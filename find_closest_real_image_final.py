import torch
import numpy as np
import sys
import os
import glob
import random
import argparse
from PIL import Image
from torchvision import transforms as TF

# --- LIGHTGLUE IMPORTS ---
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

# --- CONFIGURATION ---
COLMAP_PATH = "/scratch/schettip/landmark_detection/datasets/taj_mahal/dense/sparse"
if not os.path.exists(os.path.join(COLMAP_PATH, "images.bin")): COLMAP_PATH = os.path.join(COLMAP_PATH, "0")
IMAGE_DIR = "/scratch/schettip/landmark_detection/datasets/taj_mahal/dense/images"
MODEL_PATH = "/scratch/schettip/landmark_detection/model.pt"

# Constraints & Scoring Weights
NUM_ANCHORS = 80 
FOV_TOLERANCE_DEGREES = 25.0 # Relaxed slightly since we have LightGlue safety net
TOP_K_CANDIDATES = 10        # How many images to pass to LightGlue
FOV_PENALTY_WEIGHT = 6     # (Alpha) 1 degree of FOV difference subtracts 8 matches from the score

# --- IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try: import pycolmap; USE_PYCOLMAP=True
except: from colmap_io import read_model; USE_PYCOLMAP=False
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.models.vggt import VGGT

# --- MATH HELPERS ---
def qvec2rotmat(qvec):
    return np.array([[1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],[2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def umeyama_alignment(model_points, target_points):
    mu_m, mu_t = model_points.mean(0), target_points.mean(0)
    m_centered, t_centered = model_points - mu_m, target_points - mu_t
    U, S, Vt = np.linalg.svd(m_centered.T @ t_centered)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[-1, :] *= -1; R = Vt.T @ U.T
    s = np.trace(np.diag(S)) / np.sum(m_centered ** 2)
    t = mu_t - s * (R @ mu_m)
    return s, R, t

def rotation_distance(R1, R2):
    R_diff = R1 @ R2.T
    tr = np.trace(R_diff)
    cos_theta = np.clip((tr - 1) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def get_fov(focal_length, max_dim):
    return 2 * np.degrees(np.arctan(max_dim / (2.0 * focal_length)))

def get_colmap_data(colmap_path):
    print(f"Loading COLMAP data (Poses & Intrinsics)...")
    name2pose = {}
    if USE_PYCOLMAP:
        recon = pycolmap.Reconstruction(colmap_path)
        for _, img in recon.images.items():
            cam = recon.cameras[img.camera_id]
            f = cam.params[0]
            name2pose[img.name] = (img.cam_from_world.rotation.matrix().T, img.projection_center(), f, cam.width, cam.height)
    else:
        cameras, images, _ = read_model(colmap_path)
        for _, img in images.items():
            cam = cameras[img.camera_id]
            f = cam.params[0] 
            R_c2w = qvec2rotmat(img.qvec).T
            name2pose[img.name] = (R_c2w, -R_c2w @ np.array(img.tvec), f, cam.width, cam.height)
    return name2pose

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_img", required=True)
    parser.add_argument("--output", default="retrieved_real.jpg")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name2pose = get_colmap_data(COLMAP_PATH)
    
    # 1. Target Image Orientation
    print(f"\nAnalyzing target image: {args.target_img}")
    with Image.open(args.target_img) as t_img:
        target_is_landscape = t_img.width >= t_img.height
        orient_str = "Landscape" if target_is_landscape else "Portrait"

    # 2. Select Anchors
    all_imgs = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    random.shuffle(all_imgs)
    anchors =[]
    print(f"Selecting {orient_str} anchors...")
    for p in all_imgs:
        if os.path.basename(p) == os.path.basename(args.target_img) or os.path.basename(p) not in name2pose: continue
        try:
            with Image.open(p) as i: 
                if (i.width >= i.height) == target_is_landscape: anchors.append(p)
        except: pass
        if len(anchors) >= NUM_ANCHORS: break

    # 3. Run VGGT Inference
    batch_paths = anchors +[args.target_img]
    print(f"🚀 Running VGGT on {len(batch_paths)} images...")
    
    model = VGGT().to(device).eval()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    
    img_tensor = load_and_preprocess_images(batch_paths, mode="pad").to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model(img_tensor.unsqueeze(0))
        ext_pred, intri_pred = pose_encoding_to_extri_intri(pred["pose_enc"], (518, 518))

    # 4. Calculate Centers & FOV
    ext_np, intri_np = ext_pred.squeeze(0).float().cpu().numpy(), intri_pred.squeeze(0).float().cpu().numpy()
    vggt_centers, vggt_rots =[],[]
    for i in range(len(ext_np)):
        R, t = ext_np[i, :3, :3], ext_np[i, :3, 3]
        vggt_rots.append(R.T)
        vggt_centers.append(-R.T @ t)
    vggt_centers = np.array(vggt_centers)

    target_fov = get_fov(intri_np[-1][0, 0], 518.0) 

    # 5. Align VGGT -> COLMAP
    gt_anchor_centers = np.array([name2pose[os.path.basename(p)][1] for p in anchors])
    s, R_align, t_align = umeyama_alignment(vggt_centers[:-1], gt_anchor_centers)
    pred_target_center = s * (R_align @ vggt_centers[-1]) + t_align
    pred_target_rot = R_align @ vggt_rots[-1] 

    print(f"\n🎯 Predicted Target FOV: {target_fov:.1f} degrees")

    # =========================================================================
    # STAGE 1: GEOMETRIC PRE-FILTERING (Find Top-K Candidates)
    # =========================================================================
    print(f"\n STAGE 1: Filtering database by Geometry (Pose, FOV, Orientation)...")
    candidates =[]
    
    for name, (gt_rot, gt_center, db_f, db_w, db_h) in name2pose.items():
        if (db_w >= db_h) != target_is_landscape: continue
        
        dist = np.linalg.norm(pred_target_center - gt_center)
        angle = rotation_distance(pred_target_rot, gt_rot)
        db_fov = get_fov(db_f, max(db_w, db_h))
        fov_diff = abs(target_fov - db_fov)
        
        if angle < 60.0 and fov_diff < FOV_TOLERANCE_DEGREES:
            candidates.append({ 'name': name, 'dist': dist, 'angle': angle, 'fov_diff': fov_diff })

    if not candidates:
        print("⚠️ Strict FOV failed. Relaxing FOV constraint...")
        for name, (gt_rot, gt_center, db_f, db_w, db_h) in name2pose.items():
            if (db_w >= db_h) != target_is_landscape: continue
            dist = np.linalg.norm(pred_target_center - gt_center)
            angle = rotation_distance(pred_target_rot, gt_rot)
            if angle < 60.0:
                candidates.append({ 'name': name, 'dist': dist, 'angle': angle, 'fov_diff': abs(target_fov - get_fov(db_f, max(db_w, db_h))) })

    # Sort by 3D Distance and keep top K
    candidates.sort(key=lambda x: x['dist'])
    top_candidates = candidates[:TOP_K_CANDIDATES]
    print(f"Found {len(top_candidates)} strong geometric candidates.")

    # =========================================================================
    # STAGE 2: VISUAL RERANKING (LightGlue + FOV Penalty)
    # =========================================================================
    print(f"\n🔍 STAGE 2: Reranking Top {len(top_candidates)} using Intrinsics-Aware Composite Score...")
    
    # Load Models dynamically to save memory
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    
    # Process Target
    target_pil = Image.open(args.target_img).convert("RGB")
    target_tensor = TF.ToTensor()(target_pil).unsqueeze(0).to(device)
    with torch.no_grad(): feats_target = extractor.extract(target_tensor)

    best_match_name = None
    best_composite_score = -float('inf') 
    best_match_stats = {}
    best_raw_matches = 0

    for cand in top_candidates:
        cand_path = os.path.join(IMAGE_DIR, cand['name'])
        if not os.path.exists(cand_path): continue
        
        cand_pil = Image.open(cand_path).convert("RGB")
        cand_tensor = TF.ToTensor()(cand_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feats_cand = extractor.extract(cand_tensor)
            matches = matcher({"image0": feats_target, "image1": feats_cand})
            matches = rbd(matches)
            
        num_inliers = len(matches['matches'])
        
        # --- NEW: COMPOSITE SCORING MATH ---
        # Score = Raw Matches - (Penalty_Weight * FOV_Difference)
        composite_score = num_inliers - (FOV_PENALTY_WEIGHT * cand['fov_diff'])
        
        print(f"   -> {cand['name']}: {cand['dist']:.1f} dist | {cand['fov_diff']:.1f}° fov_err | {num_inliers} matches | Score: {composite_score:.1f}")
        
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_match_name = cand['name']
            best_match_stats = cand
            best_raw_matches = num_inliers

    # 8. Output Results
    if best_match_name is None:
        print("\n CRITICAL: No matching image could be found.")
        sys.exit()

    print(f"\n🏆 ULTIMATE MATCH FOUND: {best_match_name}")
    print(f"   Composite Score: {best_composite_score:.1f}")
    print(f"   Raw Matches:     {best_raw_matches} exact visual points")
    print(f"   Distance:        {best_match_stats['dist']:.2f} 3D units")
    print(f"   Angle Diff:      {best_match_stats['angle']:.2f} degrees")
    print(f"   FOV Diff:        {best_match_stats['fov_diff']:.1f} degrees")

    img = Image.open(os.path.join(IMAGE_DIR, best_match_name))
    img.save(args.output)
    print(f"💾 Saved perfect retrieval to: {args.output}")

if __name__ == "__main__":
    main()