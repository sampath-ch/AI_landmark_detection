import torch
import numpy as np
import sys
import os
import glob
import random
import argparse
from PIL import Image

# --- CONFIGURATION ---
COLMAP_PATH = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/sparse"
if not os.path.exists(os.path.join(COLMAP_PATH, "images.bin")): COLMAP_PATH = os.path.join(COLMAP_PATH, "0")
IMAGE_DIR = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images"
MODEL_PATH = "/scratch/schettip/landmark_detection/model.pt"

# Alignment Settings
NUM_ANCHORS = 49 

# --- IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try: import pycolmap; USE_PYCOLMAP=True
except: from colmap_io import read_model; USE_PYCOLMAP=False
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.models.vggt import VGGT

# --- MATH HELPERS ---
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def umeyama_alignment(model_points, target_points):
    """Aligns VGGT (model) to COLMAP (target)."""
    mu_m, mu_t = model_points.mean(0), target_points.mean(0)
    m_centered, t_centered = model_points - mu_m, target_points - mu_t
    
    U, S, Vt = np.linalg.svd(m_centered.T @ t_centered)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[-1, :] *= -1; R = Vt.T @ U.T
    
    # Correct scale calculation (Sum of Squares)
    s = np.trace(np.diag(S)) / np.sum(m_centered ** 2)
    t = mu_t - s * (R @ mu_m)
    return s, R, t

def rotation_distance(R1, R2):
    """Returns angle in degrees between two rotation matrices."""
    R_diff = R1 @ R2.T
    tr = np.trace(R_diff)
    cos_theta = (tr - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def get_colmap_data(colmap_path):
    print(f"📂 Loading COLMAP data from {colmap_path}...")
    name2pose = {}
    if USE_PYCOLMAP:
        recon = pycolmap.Reconstruction(colmap_path)
        for _, img in recon.images.items():
            name2pose[img.name] = (img.cam_from_world.rotation.matrix().T, img.projection_center())
    else:
        _, images, _ = read_model(colmap_path)
        for _, img in images.items():
            R_c2w = qvec2rotmat(img.qvec).T
            name2pose[img.name] = (R_c2w, -R_c2w @ np.array(img.tvec))
    return name2pose

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_img", required=True, help="Path to the input image (Real or AI)")
    parser.add_argument("--output", default="retrieved_real.jpg", help="Path to save the closest real image")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name2pose = get_colmap_data(COLMAP_PATH)
    
    # 1. Select Anchors
    all_imgs = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    random.shuffle(all_imgs)
    
    anchors = []
    print("🔍 Selecting anchors...")
    for p in all_imgs:
        if os.path.basename(p) == os.path.basename(args.target_img): continue
        if os.path.basename(p) not in name2pose: continue
        try:
            with Image.open(p) as i: 
                if i.width >= i.height: anchors.append(p)
        except: pass
        if len(anchors) >= NUM_ANCHORS: break
    
    if len(anchors) < 10: 
        print("❌ Not enough valid anchors found.")
        sys.exit()

    # 2. Run VGGT Inference
    batch_paths = anchors + [args.target_img]
    print(f"🚀 Running VGGT on {len(batch_paths)} images...")
    
    model = VGGT().to(device).eval()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    
    img_tensor = load_and_preprocess_images(batch_paths, mode="pad").to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model(img_tensor.unsqueeze(0))
        ext_pred, _ = pose_encoding_to_extri_intri(pred["pose_enc"], (518, 518))

    # 3. Calculate Centers in VGGT Space
    ext_np = ext_pred.squeeze(0).float().cpu().numpy()
    vggt_centers = []
    vggt_rots = []
    for i in range(len(ext_np)):
        R = ext_np[i, :3, :3]
        t = ext_np[i, :3, 3]
        R_c2w = R.T
        vggt_rots.append(R_c2w)
        vggt_centers.append(-R_c2w @ t)
    vggt_centers = np.array(vggt_centers)

    # 4. Align VGGT -> COLMAP using Anchors
    gt_anchor_centers = np.array([name2pose[os.path.basename(p)][1] for p in anchors])
    s, R_align, t_align = umeyama_alignment(vggt_centers[:-1], gt_anchor_centers)
    
    # Check Alignment Stability
    err = np.mean(np.linalg.norm((s*(vggt_centers[:-1]@R_align.T)+t_align) - gt_anchor_centers, axis=1))
    print(f"✅ Alignment RMSE: {err:.4f}")

    # 5. Transform Target Pose to COLMAP Space
    pred_target_center = s * (R_align @ vggt_centers[-1]) + t_align
    pred_target_rot = R_align @ vggt_rots[-1] 

    print(f"\n📍 Predicted Target Location: {pred_target_center}")

    # 6. Find Closest Real Image in Database
    print("🔎 Searching database for closest view...")
    
    best_dist = float('inf')
    best_img_name = None
    best_angle = 0.0
    
    for name, (gt_rot, gt_center) in name2pose.items():
        dist = np.linalg.norm(pred_target_center - gt_center)
        angle = rotation_distance(pred_target_rot, gt_rot)
        
        # Prioritize distance, enforce simple angle check
        if dist < best_dist and angle < 60.0:
            best_dist = dist
            best_img_name = name
            best_angle = angle

    print(f"🎉 Match Found: {best_img_name}")
    print(f"   Distance: {best_dist:.2f} units")
    print(f"   Angle Diff: {best_angle:.2f} degrees")

    # 7. Save the Retrieved Image
    match_path = os.path.join(IMAGE_DIR, best_img_name)
    if os.path.exists(match_path):
        img = Image.open(match_path)
        img.save(args.output)
        print(f"💾 Saved retrieved image to: {args.output}")
    else:
        print(f"❌ Error: Matched file {best_img_name} not found in directory.")

if __name__ == "__main__":
    main()