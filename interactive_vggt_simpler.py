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

# 1. VGGT
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

def get_vggt_pose_and_intrinsics(anchor_tensor, target_tensor):
    batch = torch.cat([anchor_tensor, target_tensor], dim=0)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        pred = vggt_model(batch.unsqueeze(0))
        ext, intri = pose_encoding_to_extri_intri(pred["pose_enc"], (518, 518))
    
    ext_np = ext.squeeze(0).float().cpu().numpy()
    intri_np = intri.squeeze(0).float().cpu().numpy()
    K = intri_np[0] 
    
    centers = []
    for i in range(2):
        R = ext_np[i, :3, :3]
        t = ext_np[i, :3, 3]
        centers.append(-R.T @ t)
        
    translation_vec = centers[1] - centers[0]
    
    if np.linalg.norm(translation_vec) > 0:
        translation_vec = translation_vec / np.linalg.norm(translation_vec)
        
    return translation_vec, K

def get_lightglue_pose_and_matches(img1_tensor, img2_tensor, K):
    with torch.no_grad():
        feats0 = extractor.extract(img1_tensor)
        feats1 = extractor.extract(img2_tensor)
        
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    
    matches = matches01['matches']
    points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
    points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()
    
    if len(points0) < 8:
        return np.zeros(3), points0, points1

    E, mask = cv2.findEssentialMat(points0, points1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return np.zeros(3), points0, points1

    _, R, t, mask = cv2.recoverPose(E, points0, points1, K, mask=mask)
    
    valid_mask = mask.flatten() > 0
    inlier_pts0 = points0[valid_mask]
    inlier_pts1 = points1[valid_mask]
    
    center = -R.T @ t.flatten()
    return center, inlier_pts0, inlier_pts1

def plot_lightglue_matches(img1_tensor, img2_tensor, pts0, pts1, save_path):
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
    
    for i in range(len(pts0)):
        x0, y0 = pts0[i]
        x1, y1 = pts1[i]
        x1 += w1
        ax.plot([x0, x1], [y0, y1], color='lime', linewidth=0.5, alpha=0.7)
        ax.plot(x0, y0, 'ro', markersize=2)
        ax.plot(x1, y1, 'ro', markersize=2)
        
    ax.set_title(f"LightGlue Inlier Matches (N={len(pts0)})", fontsize=14)
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
        masked_pil = base_pil.copy()
        
        box_label = "Baseline (No Mask)"
        if len(points) == 2:
            x0, x1 = sorted([points[0][0], points[1][0]])
            y0, y1 = sorted([points[0][1], points[1][1]])
            draw = ImageDraw.Draw(masked_pil)
            draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
            box_label = f"[{x0},{y0}] to [{x1},{y1}]"
            flush_print(f"   -> Black mask applied at box: {box_label}")

        anchor_tensor = process_img_to_tensor(anchor_pil).to(DEVICE)
        base_tensor = process_img_to_tensor(base_pil).to(DEVICE)
        masked_tensor = process_img_to_tensor(masked_pil).to(DEVICE)
        
        flush_print("2. Running VGGT Inference...")
        vggt_base_vec, K = get_vggt_pose_and_intrinsics(anchor_tensor, base_tensor)
        vggt_masked_vec, _ = get_vggt_pose_and_intrinsics(anchor_tensor, masked_tensor)
        
        flush_print("3. Running LightGlue Inference...")
        lg_base_vec, _, _ = get_lightglue_pose_and_matches(anchor_tensor, base_tensor, K)
        lg_masked_vec, lg_pts0, lg_pts1 = get_lightglue_pose_and_matches(anchor_tensor, masked_tensor, K)
        
        flush_print("4. Generating Drift Map...")
        vggt_drift = np.linalg.norm(vggt_masked_vec - vggt_base_vec)
        lg_drift = np.linalg.norm(lg_masked_vec - lg_base_vec)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(0, 0, 'ko', markersize=12, label="Origin")
        ax.plot(vggt_base_vec[0], vggt_base_vec[2], 'bo', markersize=10, label="VGGT: Base")
        ax.plot(vggt_masked_vec[0], vggt_masked_vec[2], 'bx', markersize=12, label="VGGT: Mask")
        ax.annotate('', xy=(vggt_masked_vec[0], vggt_masked_vec[2]), xytext=(vggt_base_vec[0], vggt_base_vec[2]), arrowprops=dict(arrowstyle="->", color='blue', lw=1.5))
            
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
        plot_lightglue_matches(anchor_tensor, masked_tensor, lg_pts0, lg_pts1, match_plot_path)
        
        result_text = f"Current Run -> VGGT: {vggt_drift:.4f} | LightGlue: {lg_drift:.4f} | Features: {len(lg_pts0)}"
        
        # Append to history state
        history.append([box_label, f"{vggt_drift:.4f}", f"{lg_drift:.4f}", str(len(lg_pts0))])
        
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
    gr.Markdown("# VGGT vs LightGlue: Geometric Forensics")
    
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
            
            # New History Table
            history_output = gr.Dataframe(
                headers=["Mask Coordinates", "VGGT Drift", "LightGlue Drift", "Retained Features"],
                label="Session History",
                interactive=False
            )
            
            # Side-by-side reduced images
            with gr.Row():
                mask_preview_output = gr.Image(label="Masked Preview", height=350) 
                drift_output = gr.Image(label="Pose Drift Map", height=350)
                
            match_output = gr.Image(label="LightGlue Feature Matches (Masked State)")

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
    PROXY_PATH = "/rnode/gpu027.orc.gmu.edu/20282/proxy/7860/"
    print(f"🚀 Launching via Proxy: {PROXY_PATH}")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        root_path=PROXY_PATH,  
        allowed_paths=["/scratch"] 
    )