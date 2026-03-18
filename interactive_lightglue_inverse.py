import gradio as gr
import torch
import numpy as np
import sys
import os
import time
import cv2
import matplotlib
matplotlib.use('Agg') # Force headless mode
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms as TF

# --- NEW IMPORTS FOR LIGHTGLUE ---
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

# --- CONFIGURATION ---
MODEL_PATH = "/scratch/schettip/landmark_detection/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# --- IMPORTS FOR VGGT ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# --- HELPERS ---
def flush_print(text):
    print(text)
    sys.stdout.flush()

# --- LOAD MODELS GLOBALLY ---
flush_print(f"Loading Models on {DEVICE}...")

# 1. VGGT (Now used only for Intrinsics Extraction)
vggt_model = VGGT().to(DEVICE).eval()
vggt_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

# 2. LightGlue & SuperPoint
extractor = SuperPoint(max_num_keypoints=2048).eval().to(DEVICE)
matcher = LightGlue(features='superpoint', depth_confidence=0.95, width_confidence=0.99, filter_threshold=0.9).eval().to(DEVICE)

# --- PROCESSING ---
def process_img_to_tensor(pil_img):
    w, h = pil_img.size
    target_size = 518
    scale = target_size / max(w, h)
    new_w, new_h = (int(w * scale) // 14) * 14, (int(h * scale) // 14) * 14
    resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
    t = TF.ToTensor()(resized)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), value=1.0).unsqueeze(0)

def get_vggt_intrinsics(anchor_tensor, target_tensor):
    """Refactored to only extract the Camera Intrinsics (K)."""
    batch = torch.cat([anchor_tensor, target_tensor], dim=0)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        pred = vggt_model(batch.unsqueeze(0))
        _, intri = pose_encoding_to_extri_intri(pred["pose_enc"], (518, 518))
    
    intri_np = intri.squeeze(0).float().cpu().numpy()
    K = intri_np[0] 
    return K

def get_lightglue_pose_and_matches(img1_tensor, img2_tensor, K):
    with torch.no_grad():
        feats0 = extractor.extract(img1_tensor)
        feats1 = extractor.extract(img2_tensor)
        
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    
    # These are the RAW initial correspondences picked by SuperPoint/LightGlue
    matches = matches01['matches']
    points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
    points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()
    
    if len(points0) < 8:
        # Return empty mask if not enough points
        return np.zeros(3), points0, points1, np.zeros(len(points0), dtype=bool)

    # RANSAC filters the raw points to find geometrically rigid INLIERS
    E, mask = cv2.findEssentialMat(points0, points1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return np.zeros(3), points0, points1, np.zeros(len(points0), dtype=bool)

    _, R, t, mask = cv2.recoverPose(E, points0, points1, K, mask=mask)
    
    valid_mask = mask.flatten() > 0
    center = -R.T @ t.flatten()
    
    # Return everything so we can plot both raw and inlier points
    return center, points0, points1, valid_mask

def plot_lightglue_matches(img1_tensor, img2_tensor, pts0, pts1, valid_mask, save_path):
    img1_np = img1_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img2_np = img2_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    h1, w1, _ = img1_np.shape
    h2, w2, _ = img2_np.shape
    
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.float32)
    canvas[:h1, :w1, :] = img1_np
    canvas[:h2, w1:w1+w2, :] = img2_np
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas)
    ax.axis('off')
    
    inlier_count = 0
    for i in range(len(pts0)):
        x0, y0 = pts0[i]
        x1, y1 = pts1[i]
        x1 += w1
        
        if valid_mask[i]:
            # Inliers: Bright solid green lines
            ax.plot([x0, x1], [y0, y1], color='lime', linewidth=1.0, alpha=0.9)
            ax.plot(x0, y0, 'go', markersize=3)
            ax.plot(x1, y1, 'go', markersize=3)
            inlier_count += 1
        else:
            # Outliers (Raw matches rejected by geometry): Faint dotted red lines
            ax.plot([x0, x1], [y0, y1], color='red', linewidth=0.5, alpha=0.4, linestyle=':')
            ax.plot(x0, y0, 'rx', markersize=2)
            ax.plot(x1, y1, 'rx', markersize=2)
        
    ax.set_title(f"LightGlue Matches (Raw Total: {len(pts0)} | Rigid Inliers: {inlier_count})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

# --- CLICK HANDLING LOGIC ---
def get_click_coords(clean_img, display_img, points, evt: gr.SelectData):
    base_img = clean_img if clean_img is not None else display_img
    if base_img is None: return display_img, points
    if len(points) >= 2: points = []

    points.append((evt.index[0], evt.index[1]))
    img_np = np.array(base_img.copy())
    
    for pt in points:
        cv2.circle(img_np, pt, radius=5, color=(0, 0, 255), thickness=-1)
    if len(points) == 2:
        cv2.rectangle(img_np, points[0], points[1], (0, 0, 255), thickness=3)
        
    return Image.fromarray(img_np), points

def clear_points(clean_img):
    return clean_img, []

def update_clean_img(img):
    return img

# --- MAIN ANALYSIS FUNCTION ---
def analyze(anchor_img, clean_target_img, points, history):
    try:
        flush_print("\n1. Request Received. Processing Images...")
        if history is None:
            history = []
            
        if anchor_img is None or clean_target_img is None: 
            return None, None, None, "Error: Missing images.", history, history
            
        anchor_pil = anchor_img.convert("RGB")
        base_pil = clean_target_img.convert("RGB")
        
        box_label = "Baseline (Full Image)"
        if len(points) == 2:
            x0, x1 = sorted([points[0][0], points[1][0]])
            y0, y1 = sorted([points[0][1], points[1][1]])
            
            # --- MODIFIED: Isolate Crop, Black out everything else ---
            masked_pil = Image.new("RGB", base_pil.size, (0, 0, 0))
            crop = base_pil.crop((x0, y0, x1, y1))
            masked_pil.paste(crop, (x0, y0))
            
            box_label = f"Isolated Crop: [{x0},{y0}] to [{x1},{y1}]"
            flush_print(f"   -> Image inverted masked (kept only box): {box_label}")
        else:
            masked_pil = base_pil.copy()
            flush_print("   -> No valid mask defined (Baseline Run).")

        anchor_tensor = process_img_to_tensor(anchor_pil).to(DEVICE)
        base_tensor = process_img_to_tensor(base_pil).to(DEVICE)
        masked_tensor = process_img_to_tensor(masked_pil).to(DEVICE)
        
        flush_print("2. Extracting Intrinsics via VGGT...")
        K = get_vggt_intrinsics(anchor_tensor, base_tensor)
        
        flush_print("3. Running LightGlue Inference...")
        lg_base_vec, raw0_base, raw1_base, mask_base = get_lightglue_pose_and_matches(anchor_tensor, base_tensor, K)
        lg_masked_vec, raw0_mask, raw1_mask, mask_mask = get_lightglue_pose_and_matches(anchor_tensor, masked_tensor, K)
        
        flush_print("4. Generating Drift Map...")
        lg_drift = np.linalg.norm(lg_masked_vec - lg_base_vec)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(0, 0, 'ko', markersize=12, label="Origin")
        ax.plot(lg_base_vec[0], lg_base_vec[2], 'ro', markersize=10, label="LG: Base")
        ax.plot(lg_masked_vec[0], lg_masked_vec[2], 'rx', markersize=12, label="LG: Mask")
        ax.annotate('', xy=(lg_masked_vec[0], lg_masked_vec[2]), xytext=(lg_base_vec[0], lg_base_vec[2]), arrowprops=dict(arrowstyle="->", color='red', lw=1.5, ls='--'))
            
        ax.set_title("Pose Drift Map", fontsize=12)
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        drift_plot_path = "temp_drift_plot.png"
        plt.savefig(drift_plot_path)
        plt.close()

        flush_print("5. Generating Match Visualization...")
        match_plot_path = "temp_match_plot.png"
        plot_lightglue_matches(anchor_tensor, masked_tensor, raw0_mask, raw1_mask, mask_mask, match_plot_path)
        
        inlier_count = sum(mask_mask)
        raw_count = len(raw0_mask)
        result_text = f"Current Run -> LightGlue Drift: {lg_drift:.4f} | Raw Matches: {raw_count} | Inliers: {inlier_count}"
        
        # Append to history state
        history.append([box_label, f"{lg_drift:.4f}", str(raw_count), str(inlier_count)])
        
        flush_print("Done.")
        return masked_pil, drift_plot_path, match_plot_path, result_text, history, history
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, f"Error: {str(e)}", history, history

def clear_history():
    return [], []

# --- UI ---
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# LightGlue Geometric Forensics (Isolated Crop Analysis)")
    
    clean_target_state = gr.State(None)
    points_state = gr.State([])
    history_state = gr.State([])
    
    with gr.Row():
        # LEFT COLUMN (Inputs)
        with gr.Column(scale=1):
            anchor_input = gr.Image(label="1. Anchor Image (Reference)", type="pil")
            target_input = gr.Image(label="2. Target Image (Click 2 points for bounding box)", type="pil", interactive=True)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Image Selection")
                clear_hist_btn = gr.Button("Clear History Log")
                analyze_btn = gr.Button("Analyze Rigid Geometry", variant="primary")
            
        # RIGHT COLUMN (Outputs)
        with gr.Column(scale=1):
            text_output = gr.Textbox(label="Numerical Results", lines=1)
            
            # Updated History Table headers to track raw vs inliers
            history_output = gr.Dataframe(
                headers=["Region Analyzed", "LG Drift", "Raw Matches", "Inliers"],
                label="Session History",
                interactive=False
            )
            
            # Side-by-side reduced images
            with gr.Row():
                mask_preview_output = gr.Image(label="Isolated Crop Preview", height=350) 
                drift_output = gr.Image(label="Pose Drift Map", height=350)
                
            match_output = gr.Image(label="LightGlue Feature Matches (Red = Raw/Outliers, Green = Inliers)")

    # Event Wiring
    target_input.upload(update_clean_img, inputs=[target_input], outputs=[clean_target_state])
    target_input.select(get_click_coords, inputs=[clean_target_state, target_input, points_state], outputs=[target_input, points_state])
    clear_btn.click(clear_points, inputs=[clean_target_state], outputs=[target_input, points_state])
    clear_hist_btn.click(clear_history, inputs=None, outputs=[history_state, history_output])

    analyze_btn.click(
        analyze, 
        inputs=[anchor_input, clean_target_state, points_state, history_state], 
        outputs=[mask_preview_output, drift_output, match_output, text_output, history_state, history_output], 
        queue=False
    )

if __name__ == "__main__":
    PROXY_PATH = "/rnode/gpu030.orc.gmu.edu/37157/proxy/7860/"
    print(f"🚀 Launching via Proxy: {PROXY_PATH}")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        root_path=PROXY_PATH,  
        allowed_paths=["/scratch"] 
    )