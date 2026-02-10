import torch
import os
import numpy as np
import sys
import glob
import random
from PIL import Image
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
COLMAP_PATH = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/sparse"
IMAGE_DIR = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images"
MODEL_PATH = "/scratch/schettip/landmark_detection/model.pt"

# The specific real image we want to validate against
TARGET_IMAGE_NAME = "00289298_7642283248.jpg"

# Fix random seed for consistent context selection
RANDOM_SEED = 42
NUM_ANCHORS = 49

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

# --- HELPERS ---

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
    """Computes Sim(3) alignment."""
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

def rotation_error(R1, R2):
    """Calculates angular difference in degrees between two rotation matrices."""
    # R_diff = R1 @ R2^T
    # trace = 1 + 2 cos(theta)
    R_diff = R1 @ R2.T
    tr = np.trace(R_diff)
    cos_theta = (tr - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)

def get_colmap_data(colmap_path):
    print(f"📂 Loading COLMAP from {colmap_path}...")
    name2pose = {} # Stores (R_c2w, Center)
    
    if USE_PYCOLMAP:
        recon = pycolmap.Reconstruction(colmap_path)
        for _, img in recon.images.items():
            R_w2c = img.cam_from_world.rotation.matrix()
            t_w2c = img.cam_from_world.translation
            center = img.projection_center()
            name2pose[img.name] = (R_w2c.T, center) # Store C2W Rotation and Center
    else:
        _, images, _ = read_model(colmap_path)
        for _, img in images.items():
            R_w2c = qvec2rotmat(img.qvec)
            t_w2c = np.array(img.tvec)
            R_c2w = R_w2c.T
            center = -R_c2w @ t_w2c
            name2pose[img.name] = (R_c2w, center)
            
    return name2pose

# --- MAIN ---

# 1. Setup Data
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

name2pose = get_colmap_data(COLMAP_PATH)
all_jpgs = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))

# Filter anchors
valid_anchors = []
for path in all_jpgs:
    fname = os.path.basename(path)
    if fname == TARGET_IMAGE_NAME: continue
    if fname not in name2pose: continue
    # Skip portrait
    try:
        with Image.open(path) as img:
            if img.width < img.height: continue
    except: continue
    valid_anchors.append(path)

if len(valid_anchors) < NUM_ANCHORS:
    print(f"❌ Not enough anchors! Found {len(valid_anchors)}")
    sys.exit()

anchor_paths = random.sample(valid_anchors, NUM_ANCHORS)
target_path_full = os.path.join(IMAGE_DIR, TARGET_IMAGE_NAME)
batch_paths = anchor_paths + [target_path_full]

print(f"🔹 Running VGGT on {len(batch_paths)} images...")

# 2. Run VGGT
model = VGGT()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model = model.to(DEVICE).eval()

images_tensor = load_and_preprocess_images(batch_paths, mode="pad").to(DEVICE)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images_tensor.unsqueeze(0))
        pose_enc = predictions["pose_enc"] 
        # Extrinsics are World-To-Camera (OpenCV)
        extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])

extrinsic_np = extrinsic.squeeze(0).float().cpu().numpy()

# 3. Process VGGT Data
vggt_centers = []
vggt_rots_c2w = []

for i in range(len(extrinsic_np)):
    R_w2c = extrinsic_np[i, :3, :3]
    t_w2c = extrinsic_np[i, :3, 3]
    R_c2w = R_w2c.T
    C = -R_c2w @ t_w2c
    vggt_centers.append(C)
    vggt_rots_c2w.append(R_c2w)

vggt_centers = np.array(vggt_centers)
vggt_rots_c2w = np.array(vggt_rots_c2w)

# Split Context / Target
vggt_anchor_centers = vggt_centers[:-1]
vggt_target_center = vggt_centers[-1]
vggt_target_rot_c2w = vggt_rots_c2w[-1]

# Get Ground Truth Data
gt_anchor_centers = np.array([name2pose[os.path.basename(p)][1] for p in anchor_paths])
gt_target_rot_c2w, gt_target_center = name2pose[TARGET_IMAGE_NAME]

# 4. Alignment (Sim3 on Centers)
s, R_align, t_align = umeyama_alignment(vggt_anchor_centers, gt_anchor_centers)

# Sanity Check on Scale
anchor_err = np.mean(np.linalg.norm((s * (vggt_anchor_centers @ R_align.T) + t_align) - gt_anchor_centers, axis=1))
print(f"\n✅ Anchor Alignment Error: {anchor_err:.4f} (Should be low, e.g. < 5.0)")

# 5. PREDICT TARGET POSE (Base Assumption)
# Center
pred_center = s * (R_align @ vggt_target_center) + t_align
center_error = np.linalg.norm(pred_center - gt_target_center)

# Rotation (Base): Apply alignment rotation to VGGT rotation
# R_new = R_align * R_old
pred_rot_base = R_align @ vggt_target_rot_c2w

print(f"\n🎯 TARGET IMAGE ANALYSIS ({TARGET_IMAGE_NAME})")
print(f"   Position Error: {center_error:.4f} units")

# 6. TEST ROTATION PERMUTATIONS
print("\n--- ROTATION HYPOTHESIS TESTING ---")
print("Comparing Predicted vs Ground Truth (COLMAP C2W)")

# Hypothesis A: Direct Match (VGGT output is standard OpenCV C2W)
err_a = rotation_error(pred_rot_base, gt_target_rot_c2w)
print(f"A. Direct Match:                {err_a:.2f}°")

# Hypothesis B: Coordinate Flip (OpenCV -> OpenGL conversion needed?)
# Usually applied on the right: R_new = R_base @ FlipYZ
FLIP_YZ = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
pred_rot_flip = pred_rot_base @ FLIP_YZ
err_b = rotation_error(pred_rot_flip, gt_target_rot_c2w)
print(f"B. With YZ Flip (OpenGL):       {err_b:.2f}°")

# Hypothesis C: Transpose/Inverse (Maybe VGGT output was actually C2W?)
# If VGGT output was C2W, then we inverted it twice. Let's try inverting it back.
R_w2c_vggt_orig = vggt_target_rot_c2w.T 
pred_rot_inverted = R_align @ R_w2c_vggt_orig
err_c = rotation_error(pred_rot_inverted, gt_target_rot_c2w)
print(f"C. Inverted Input:              {err_c:.2f}°")

# Hypothesis D: Pre-Alignment Flip
# Maybe the flip happens BEFORE alignment?
pred_rot_preflip = R_align @ (vggt_target_rot_c2w @ FLIP_YZ)
err_d = rotation_error(pred_rot_preflip, gt_target_rot_c2w)
print(f"D. Pre-alignment Flip:          {err_d:.2f}°")

print("\n--- CONCLUSION ---")
best_err = min(err_a, err_b, err_c, err_d)
if best_err < 10.0:
    if best_err == err_a:
        print("✅ MATCH FOUND: 'Direct Match' is correct. Do NOT flip coordinates.")
    elif best_err == err_b:
        print("✅ MATCH FOUND: 'OpenGL Flip' is correct. You MUST apply YZ flip.")
    elif best_err == err_c:
        print("✅ MATCH FOUND: 'Inverted Input'. VGGT output format assumption was wrong.")
    else:
        print("✅ MATCH FOUND: 'Pre-flip'.")
else:
    print("❌ NO MATCH: Even the rotation is wrong (>10° error).")
    print("   This implies the VGGT prediction itself is garbage, or the anchors are bad.")