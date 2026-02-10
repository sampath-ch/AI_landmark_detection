import torch
import os
import numpy as np
import sys
import glob
import random
import json
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
COLMAP_PATH = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/sparse"
IMAGE_DIR = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images"
MODEL_PATH = "/scratch/schettip/landmark_detection/model.pt"

# The Single "AI" Image you want to test (Provide full path)
TARGET_IMAGE_PATH = "/scratch/schettip/landmark_detection/ai_generated.jpeg" 

# Context Config
NUM_ANCHORS = 50  # More anchors = Better scale stability

# Check COLMAP path structure
if not os.path.exists(os.path.join(COLMAP_PATH, "images.bin")):
    COLMAP_PATH = os.path.join(COLMAP_PATH, "0")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

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
    print(f"❌ Import Error: {e}")
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
    """
    Computes Sim(3) alignment: target = s * R * model + t
    Corrected to use Sum of Squares for variance.
    """
    mu_m = model_points.mean(0)
    mu_t = target_points.mean(0)
    m_centered = model_points - mu_m
    t_centered = target_points - mu_t

    H = m_centered.T @ t_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection check
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Variance (Sum of Squared Errors)
    var_m = np.sum(m_centered ** 2)
    
    # Scale
    s = np.trace(np.diag(S)) / var_m
    
    # Translation
    t = mu_t - s * (R @ mu_m)
    return s, R, t

def get_colmap_data(colmap_path):
    print(f"📂 Loading COLMAP from {colmap_path}...")
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

# --- MAIN PIPELINE ---

# 1. Load Anchors (Real Images)
name2center = get_colmap_data(COLMAP_PATH)
all_jpgs = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
target_filename = os.path.basename(TARGET_IMAGE_PATH)

valid_anchors = []
for path in all_jpgs:
    fname = os.path.basename(path)
    if fname == target_filename: continue # Don't include target in anchors
    if fname not in name2center: continue
    
    # Filter Portrait images to ensure stable context
    try:
        with Image.open(path) as img:
            if img.width < img.height: continue
    except: continue
        
    valid_anchors.append(path)

# Select Anchors
if len(valid_anchors) > NUM_ANCHORS:
    anchor_paths = random.sample(valid_anchors, NUM_ANCHORS)
else:
    anchor_paths = valid_anchors
    print(f"⚠️ Warning: Only found {len(anchor_paths)} valid anchors.")

print(f"🔹 Using {len(anchor_paths)} Anchor Images + 1 Target Image")

# Get Anchor GT Centers
gt_anchor_centers = np.array([name2center[os.path.basename(p)] for p in anchor_paths])

# 2. Run VGGT (Batch = Anchors + Target)
# Put target at the END
batch_paths = anchor_paths + [TARGET_IMAGE_PATH]

print("🚀 Running VGGT Inference...")
model = VGGT()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model = model.to(DEVICE).eval()

# CRITICAL: mode="pad" handles aspect ratios correctly
images_tensor = load_and_preprocess_images(batch_paths, mode="pad").to(DEVICE)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # (1, N, 3, 518, 518)
        predictions = model(images_tensor.unsqueeze(0))
        pose_enc = predictions["pose_enc"] 
        # Get Extrinsics (World -> Cam, OpenCV)
        extrinsic, intrinsic_pred = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])

# Extract Data
extrinsic_np = extrinsic.squeeze(0).float().cpu().numpy() # (N, 3, 4)
intrinsic_np = intrinsic_pred.squeeze(0).float().cpu().numpy() # (N, 3, 3)

# Calculate VGGT Centers
vggt_centers = []
for i in range(len(extrinsic_np)):
    R = extrinsic_np[i, :3, :3]
    t = extrinsic_np[i, :3, 3]
    vggt_centers.append(-R.T @ t)
vggt_centers = np.array(vggt_centers)

# Split Context vs Target
vggt_anchor_centers = vggt_centers[:-1]
vggt_target_center  = vggt_centers[-1]
vggt_target_extrinsic = extrinsic_np[-1] # OpenCV World-to-Cam
vggt_target_intrinsic = intrinsic_np[-1] # For 518x518 Padded

# 3. Calculate Sim3 Alignment (Anchors Only)
s, R_align, t_align = umeyama_alignment(vggt_anchor_centers, gt_anchor_centers)

# Check Stability
aligned_anchors = (s * (vggt_anchor_centers @ R_align.T) + t_align)
rmse = np.sqrt(np.mean(np.sum((aligned_anchors - gt_anchor_centers)**2, axis=1)))
print(f"✅ Alignment RMSE on Anchors: {rmse:.4f}")

if rmse > 5.0:
    print("⚠️ WARNING: Alignment is unstable. Render might be incorrect.")

# ---------------------------------------------------------
# TRAP HANDLING & EXPORT
# ---------------------------------------------------------

# --- TRAP 2: Re-assemble Rotation & Position ---
# We have World-to-Cam. Invert to get Cam-to-World (Pose)
# Pose = [ R_c2w | C ]
R_w2c_vggt = vggt_target_extrinsic[:3, :3]
R_c2w_vggt = R_w2c_vggt.T

# Rotate the Orientation
R_c2w_new = R_align @ R_c2w_vggt

# Translate and Scale the Center
C_new = s * (R_align @ vggt_target_center) + t_align

# Construct 4x4 Pose Matrix (OpenCV Convention)
c2w_opencv = np.eye(4)
c2w_opencv[:3, :3] = R_c2w_new
c2w_opencv[:3, 3] = C_new

# --- TRAP 1: Coordinate Flip (OpenCV -> OpenGL) ---
# OpenCV: Right, Down, Forward
# OpenGL: Right, Up, Back
# Matrix to flip Y and Z axes
FLIP_MAT = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])

# Apply flip on the right (local axes)
c2w_opengl = c2w_opencv @ FLIP_MAT

# --- TRAP 3: Intrinsics & Principal Point ---
# We need to output parameters for the ORIGINAL image resolution, not the 518x518 padded one.
with Image.open(TARGET_IMAGE_PATH) as img:
    orig_w, orig_h = img.size

# VGGT worked on a 518x518 padded image.
# Calculate how much scaling/padding happened.
max_dim = max(orig_w, orig_h)
scale = 518.0 / max_dim

# Get VGGT predicted Focal Length (in 518 pixel space)
# Assuming fx ≈ fy
focal_518 = vggt_target_intrinsic[0, 0]

# Scale Focal Length back to original resolution
focal_orig = focal_518 / scale

# Principal Point: Assume center of original image
# (VGGT predicts center of 518 image, which maps to center of original image if padding was symmetric)
cx_orig = orig_w / 2.0
cy_orig = orig_h / 2.0

print("\n--- RENDER PARAMETERS ---")
print(f"Target Image: {target_filename}")
print(f"Original Res: {orig_w} x {orig_h}")
print(f"Focal Length: {focal_orig:.2f}")
print(f"Pose (OpenGL):\n{c2w_opengl}")

# --- EXPORT TO JSON ---
output_data = {
    "file_path": TARGET_IMAGE_PATH,
    "transform_matrix": c2w_opengl.tolist(), # 4x4
    "fl_x": focal_orig,
    "fl_y": focal_orig,
    "cx": cx_orig,
    "cy": cy_orig,
    "w": orig_w,
    "h": orig_h,
    "camera_model": "PINHOLE"
}

out_json_path = "render_pose.json"
with open(out_json_path, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"\n💾 Saved render pose to: {out_json_path}")
print("You can now load this JSON into your renderer.")