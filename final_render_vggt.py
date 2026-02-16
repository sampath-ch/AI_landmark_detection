import torch
import json
import imageio
import numpy as np
import argparse
import sys
import os
import glob
import time
import random
from PIL import Image
from plyfile import PlyData

start_time = time.perf_counter()

# --- CONFIG ---
COLMAP_PATH = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/sparse"
if not os.path.exists(os.path.join(COLMAP_PATH, "images.bin")): COLMAP_PATH = os.path.join(COLMAP_PATH, "0")
IMAGE_DIR = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images"
MODEL_PATH = "/scratch/schettip/landmark_detection/model.pt"
NUM_ANCHORS = 49

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try: from gsplat.rendering import rasterization
except: print("❌ Install gsplat"); sys.exit()
try: import pycolmap; USE_PYCOLMAP=True
except: from colmap_io import read_model; USE_PYCOLMAP=False
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.models.vggt import VGGT

# --- MATH ---
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def umeyama_alignment(model_points, target_points):
    mu_m, mu_t = model_points.mean(0), target_points.mean(0)
    m_centered, t_centered = model_points - mu_m, target_points - mu_t
    U, S, Vt = np.linalg.svd(m_centered.T @ t_centered)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[-1, :] *= -1; R = Vt.T @ U.T
    s = np.trace(np.diag(S)) / np.sum(m_centered ** 2)
    t = mu_t - s * (R @ mu_m)
    return s, R, t

def get_colmap_data(colmap_path):
    print(f"📂 Loading COLMAP...")
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

def load_ply(path):
    print(f"☁️ Loading PLY...")
    plydata = PlyData.read(path)
    opacities_raw = plydata.elements[0]["opacity"]
    mask = opacities_raw > -3.0 # Filter noise
    
    xyz = np.stack((plydata.elements[0]["x"], plydata.elements[0]["y"], plydata.elements[0]["z"]), axis=1)[mask]
    features_dc = np.stack([plydata.elements[0]["f_dc_0"], plydata.elements[0]["f_dc_1"], plydata.elements[0]["f_dc_2"]], axis=1)[mask]
    
    scale_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")], key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for i, name in enumerate(scale_names): scales[:, i] = plydata.elements[0][name][mask]
        
    rot_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")], key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for i, name in enumerate(rot_names): rots[:, i] = plydata.elements[0][name][mask]

    return {
        "means": torch.tensor(xyz, dtype=torch.float32, device="cuda"),
        "scales": torch.tensor(scales, dtype=torch.float32, device="cuda"),
        "quats": torch.tensor(rots, dtype=torch.float32, device="cuda"),
        "opacities": torch.tensor(opacities_raw[mask][..., np.newaxis], dtype=torch.float32, device="cuda"),
        "sh0": torch.tensor(features_dc, dtype=torch.float32, device="cuda").unsqueeze(1),
    }

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", required=True)
    parser.add_argument("--target_img", required=True, help="Full path to target image")
    parser.add_argument("--output", default="final_result.png")
    args = parser.parse_args()

    # 1. SETUP VGGT
    device = "cuda"
    name2pose = get_colmap_data(COLMAP_PATH)
    
    # Select Anchors
    valid_anchors = [p for p in glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) 
                     if os.path.basename(p) in name2pose and os.path.basename(p) != os.path.basename(args.target_img)]
    
    # Filter Portrait Anchors
    anchors = []
    for p in valid_anchors:
        try: 
            with Image.open(p) as i: 
                if i.width >= i.height: anchors.append(p)
        except: pass
        if len(anchors) >= NUM_ANCHORS: break
            
    if len(anchors) < 10: print("❌ Too few anchors"); sys.exit()
    
    batch_paths = anchors + [args.target_img]
    
    print(f"🚀 Inference on {len(batch_paths)} images...")
    model = VGGT().to(device).eval()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    
    # 2. INFERENCE
    img_tensor = load_and_preprocess_images(batch_paths, mode="pad").to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model(img_tensor.unsqueeze(0))
        ext_pred, int_pred = pose_encoding_to_extri_intri(pred["pose_enc"], (518, 518))

    # 3. ALIGNMENT
    ext_np = ext_pred.squeeze(0).float().cpu().numpy()
    vggt_centers = []
    for i in range(len(ext_np)):
        vggt_centers.append(-ext_np[i, :3, :3].T @ ext_np[i, :3, 3])
    vggt_centers = np.array(vggt_centers)
    
    gt_anchor_centers = np.array([name2pose[os.path.basename(p)][1] for p in anchors])
    
    s, R_align, t_align = umeyama_alignment(vggt_centers[:-1], gt_anchor_centers)
    
    # Validate Alignment
    err = np.mean(np.linalg.norm((s*(vggt_centers[:-1]@R_align.T)+t_align) - gt_anchor_centers, axis=1))
    print(f"✅ Alignment Error: {err:.4f}")

    # 4. TRANSFORM TARGET POSE
    # Direct Match: R_new = R_align * R_old
    # VGGT R is W2C. We need C2W for logic, then back to W2C for Renderer.
    # Actually, R_align rotates the WORLD frame. 
    # If Pose = [R_c2w | C], then New_Pose = [R_align @ R_c2w | s*R_align@C + t]
    
    R_w2c_vggt = ext_np[-1, :3, :3]
    R_c2w_vggt = R_w2c_vggt.T
    
    R_c2w_final = R_align @ R_c2w_vggt
    C_final = s * (R_align @ vggt_centers[-1]) + t_align

    # --- FIX FOR "BROWN FOG" ---
    # The AI often places cameras too low (inside the floor).
    # We move the camera "UP" relative to its own orientation.
    # In OpenCV, "Up" is the NEGATIVE Y axis (Column 1 of Rotation Matrix).
    
    # Adjustable parameter: How much to lift (in COLMAP units)
    # Start with 0.5 or 1.0. If still foggy, try 2.0.
    LIFT_AMOUNT = 0.0 
    
    camera_up_vector = -R_c2w_final[:, 1] 
    C_final = C_final + (camera_up_vector * LIFT_AMOUNT)
    
    print(f"🧗 Lifting camera by {LIFT_AMOUNT} units to clear ground artifacts...")
    
    # Back to W2C for Rasterizer (No flips!)
    w2c = np.eye(4)
    w2c[:3, :3] = R_c2w_final.T
    w2c[:3, 3] = -R_c2w_final.T @ C_final
    
    viewmat = torch.tensor(w2c, dtype=torch.float32, device=device).unsqueeze(0)

    # 5. SCALE INTRINSICS
    focal_518 = int_pred[0, -1, 0, 0].item()
    with Image.open(args.target_img) as img: W, H = img.size
    scale_factor = 518.0 / max(W, H)
    fx = fy = focal_518 / scale_factor
    cx, cy = W/2, H/2
    
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)

    # 6. RENDER
    gaussians = load_ply(args.ply_path)
    rgb = gaussians["sh0"].squeeze(1) * 0.28209 + 0.5
    bg = torch.zeros((1, 3), dtype=torch.float32, device=device)

    colors, _, _ = rasterization(
        means=gaussians["means"], quats=gaussians["quats"], scales=torch.exp(gaussians["scales"]),
        opacities=torch.sigmoid(gaussians["opacities"]).squeeze(-1), colors=rgb,
        viewmats=viewmat, Ks=K, width=W, height=H, backgrounds=bg, packed=False
    )
    
    img = colors[0].clamp(0, 1).cpu().numpy()
    imageio.imwrite(args.output, (img*255).astype(np.uint8))
    print(f"🎉 Saved: {args.output}")
    end_time = time.perf_counter()

    # Calculate and print the duration
    duration = end_time - start_time
    print(f"Execution time: {duration:.4f} seconds")


if __name__ == "__main__":
    main()