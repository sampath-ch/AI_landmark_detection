import torch
import numpy as np
import argparse
import sys
import os
import glob
import time
import random
import imageio
from PIL import Image
from plyfile import PlyData

start_time = time.perf_counter()

# --- IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.models.vggt import VGGT
    try:
        import pycolmap
        USE_PYCOLMAP = True
    except ImportError:
        from colmap_io import read_model
        USE_PYCOLMAP = False
except ImportError as e:
    print(f"❌ VGGT Import Error: {e}")
    sys.exit()

try: 
    from gsplat.rendering import rasterization
except ImportError: 
    print("❌ gsplat Import Error: Please install gsplat")
    sys.exit()


# --- MATH HELPERS ---
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def umeyama_alignment(model_points, target_points):
    """Computes Sim(3) alignment: target = s * R * model + t"""
    mu_m = model_points.mean(0)
    mu_t = target_points.mean(0)
    m_centered = model_points - mu_m
    t_centered = target_points - mu_t

    H = m_centered.T @ t_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var_m = np.sum(m_centered ** 2)
    s = np.trace(np.diag(S)) / var_m
    t = mu_t - s * (R @ mu_m)
    return s, R, t

def get_colmap_data(colmap_path):
    print(f" Loading COLMAP from {colmap_path}...")
    name2center = {}
    if USE_PYCOLMAP:
        recon = pycolmap.Reconstruction(colmap_path)
        for _, img in recon.images.items():
            name2center[img.name] = img.projection_center()
    else:
        _, images, _ = read_model(colmap_path)
        for _, img in images.items():
            R = qvec2rotmat(img.qvec)
            t = np.array(img.tvec)
            name2center[img.name] = -R.T @ t
    return name2center

def load_ply(path, device):
    print(f" Loading PLY from {path}...")
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
        "means": torch.tensor(xyz, dtype=torch.float32, device=device),
        "scales": torch.tensor(scales, dtype=torch.float32, device=device),
        "quats": torch.tensor(rots, dtype=torch.float32, device=device),
        "opacities": torch.tensor(opacities_raw[mask][..., np.newaxis], dtype=torch.float32, device=device),
        "sh0": torch.tensor(features_dc, dtype=torch.float32, device=device).unsqueeze(1),
    }

# --- MAIN PIPELINE ---
def main():
    parser = argparse.ArgumentParser(description="End-to-end VGGT Pose Extraction and 3DGS Rendering")
    parser.add_argument("--target_img", required=True, help="Full path to the AI-generated target image")
    parser.add_argument("--ply_path", required=True, help="Path to the 3DGS .ply file")
    parser.add_argument("--colmap_path", required=True, help="Path to COLMAP sparse directory")
    parser.add_argument("--image_dir", required=True, help="Path to COLMAP original images directory")
    parser.add_argument("--model_path", required=True, help="Path to VGGT model.pt")
    parser.add_argument("--output", default="final_result.png", help="Output filename")
    parser.add_argument("--num_anchors", type=int, default=50, help="Number of anchor images to use")
    parser.add_argument("--lift_amount", type=float, default=0.0, help="Amount to lift camera on Y-axis to fix ground fog")
    args = parser.parse_args()

    # Hardware Setup
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Verify COLMAP path
    colmap_dir = args.colmap_path
    if not os.path.exists(os.path.join(colmap_dir, "images.bin")):
        colmap_dir = os.path.join(colmap_dir, "0")

    # ==========================================
    # STAGE 1: VGGT POSE EXTRACTION & ALIGNMENT
    # ==========================================
    name2center = get_colmap_data(colmap_dir)
    all_jpgs = glob.glob(os.path.join(args.image_dir, "*.jpg"))
    target_filename = os.path.basename(args.target_img)

    valid_anchors = []
    for path in all_jpgs:
        fname = os.path.basename(path)
        if fname == target_filename: continue
        if fname not in name2center: continue
        
        try:
            with Image.open(path) as img:
                if img.width < img.height: continue
        except: continue
        valid_anchors.append(path)

    if len(valid_anchors) > args.num_anchors:
        anchor_paths = random.sample(valid_anchors, args.num_anchors)
    else:
        anchor_paths = valid_anchors
        print(f" Warning: Only found {len(anchor_paths)} valid anchors.")

    print(f"🔹 Using {len(anchor_paths)} Anchor Images + 1 Target Image")
    gt_anchor_centers = np.array([name2center[os.path.basename(p)] for p in anchor_paths])
    batch_paths = anchor_paths + [args.target_img]

    print(" Running VGGT Inference...")
    model = VGGT()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(DEVICE).eval()

    images_tensor = load_and_preprocess_images(batch_paths, mode="pad").to(DEVICE)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images_tensor.unsqueeze(0))
            pose_enc = predictions["pose_enc"] 
            extrinsic, intrinsic_pred = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])

    # Extract Memory
    extrinsic_np = extrinsic.squeeze(0).float().cpu().numpy()
    intrinsic_np = intrinsic_pred.squeeze(0).float().cpu().numpy()

    # Calculate Centers
    vggt_centers = []
    for i in range(len(extrinsic_np)):
        R = extrinsic_np[i, :3, :3]
        t = extrinsic_np[i, :3, 3]
        vggt_centers.append(-R.T @ t)
    vggt_centers = np.array(vggt_centers)

    vggt_anchor_centers = vggt_centers[:-1]
    vggt_target_center  = vggt_centers[-1]
    vggt_target_extrinsic = extrinsic_np[-1] 
    vggt_target_intrinsic = intrinsic_np[-1] 

    # Alignment
    s, R_align, t_align = umeyama_alignment(vggt_anchor_centers, gt_anchor_centers)
    aligned_anchors = (s * (vggt_anchor_centers @ R_align.T) + t_align)
    rmse = np.sqrt(np.mean(np.sum((aligned_anchors - gt_anchor_centers)**2, axis=1)))
    print(f" Alignment RMSE on Anchors: {rmse:.4f}")

    if rmse > 5.0:
        print(" WARNING: Alignment is unstable.")

    # ==========================================
    # STAGE 2: PREPARE RENDER POSE & INTRINSICS
    # ==========================================
    # Re-assemble Rotation & Position (World-to-Cam -> Cam-to-World)
    R_w2c_vggt = vggt_target_extrinsic[:3, :3]
    R_c2w_vggt = R_w2c_vggt.T

    R_c2w_new = R_align @ R_c2w_vggt
    C_new = s * (R_align @ vggt_target_center) + t_align

    # Apply LIFT_AMOUNT to clear ground fog
    if args.lift_amount != 0.0:
        camera_up_vector = -R_c2w_new[:, 1] # OpenCV 'Up' is negative Y
        C_new = C_new + (camera_up_vector * args.lift_amount)
        print(f" Lifting camera by {args.lift_amount} units...")

    # Convert back to World-to-Camera for the Rasterizer
    w2c = np.eye(4)
    w2c[:3, :3] = R_c2w_new.T
    w2c[:3, 3] = -R_c2w_new.T @ C_new
    viewmat = torch.tensor(w2c, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    # Intrinsics Scaling
    with Image.open(args.target_img) as img:
        W, H = img.size

    max_dim = max(W, H)
    scale = 518.0 / max_dim

    focal_518 = vggt_target_intrinsic[0, 0]
    fx = fy = focal_518 / scale
    cx, cy = W / 2.0, H / 2.0

    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=DEVICE).unsqueeze(0)

    # ==========================================
    # STAGE 3: RASTERIZATION
    # ==========================================
    gaussians = load_ply(args.ply_path, DEVICE)
    rgb = gaussians["sh0"].squeeze(1) * 0.28209 + 0.5
    bg = torch.zeros((1, 3), dtype=torch.float32, device=DEVICE)

    print(f" Rendering at {W}x{H} with focal length {fx:.2f}...")
    colors, _, _ = rasterization(
        means=gaussians["means"], quats=gaussians["quats"], scales=torch.exp(gaussians["scales"]),
        opacities=torch.sigmoid(gaussians["opacities"]).squeeze(-1), colors=rgb,
        viewmats=viewmat, Ks=K, width=W, height=H, backgrounds=bg, packed=False
    )
    
    img = colors[0].clamp(0, 1).cpu().numpy()
    imageio.imwrite(args.output, (img*255).astype(np.uint8))
    print(f" Saved output to: {args.output}")

    duration = time.perf_counter() - start_time
    print(f" Total Execution time: {duration:.4f} seconds")

if __name__ == "__main__":
    main()

"""
python render_ai_pose_2.py \
    --target_img "/scratch/schettip/landmark_detection/ai_images/taj_mahal_bombed.jpeg" \
    --ply_path "/scratch/schettip/landmark_detection/external_repos/gaussian-splatting/output/46e5ee28-e/point_cloud/iteration_30000/point_cloud.ply" \
    --colmap_path "/scratch/schettip/landmark_detection/datasets/taj_mahal/dense/sparse" \
    --image_dir "/scratch/schettip/landmark_detection/datasets/taj_mahal/dense/images" \
    --model_path "/scratch/schettip/landmark_detection/model.pt" \
    --output "final_result.png" \
    --lift_amount 0.2
"""