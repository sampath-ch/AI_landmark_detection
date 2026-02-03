import torch
import os
import numpy as np
import sys
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import procrustes

# --- CONFIGURATION ---
COLMAP_PATH = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/sparse"
IMAGE_DIR = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images"
MODEL_PATH = "/scratch/schettip/landmark_detection/model.pt"

# Sampling Config
TOTAL_IMAGES = 50
NUM_CONTEXT = 45  # Anchors
NUM_TARGETS = 5   # Held-out Test

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
    """Computes Sim(3) alignment: s * R * model + t = target"""
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

    # var_m = np.var(model_points, axis=0).sum()
    var_m = np.sum(m_centered ** 2)
    s = np.trace(np.diag(S)) / var_m
    t = mu_t - s * (R @ mu_m)
    return s, R, t

def get_colmap_data(colmap_path):
    """Loads all valid image names and centers from COLMAP."""
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

# --- 1. SELECTION LOGIC ---
name2center = get_colmap_data(COLMAP_PATH)
all_jpgs = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))

valid_candidates = []
print(f"🔍 Filtering {len(all_jpgs)} images (Checking for Landscape & COLMAP existence)...")

for path in all_jpgs:
    fname = os.path.basename(path)
    
    # 1. Must exist in COLMAP
    if fname not in name2center:
        continue
        
    # 2. Must not be Portrait (Width >= Height)
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w < h: # Skip portrait
                continue
    except:
        continue
        
    valid_candidates.append(path)

print(f"✅ Found {len(valid_candidates)} valid landscape candidates.")

if len(valid_candidates) < TOTAL_IMAGES:
    print(f"❌ Not enough images! Needed {TOTAL_IMAGES}, found {len(valid_candidates)}.")
    sys.exit()

# Random Selection
selected_paths = random.sample(valid_candidates, TOTAL_IMAGES)
context_paths = selected_paths[:NUM_CONTEXT]
target_paths = selected_paths[NUM_CONTEXT:]

print(f"🔹 Selected {NUM_CONTEXT} Context Images (Anchors)")
print(f"🔹 Selected {NUM_TARGETS} Target Images (Held-out)")

# Prepare Ground Truth arrays in order
gt_centers = []
for p in selected_paths:
    gt_centers.append(name2center[os.path.basename(p)])
gt_centers = np.array(gt_centers)

# --- 2. RUN VGGT INFERENCE ---
print("🚀 Running VGGT Inference (Batch Size 30)...")
model = VGGT()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model = model.to(DEVICE).eval()

# mode="pad" is critical for robustness
images_tensor = load_and_preprocess_images(selected_paths, mode="pad").to(DEVICE)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # (1, 30, 3, 518, 518)
        predictions = model(images_tensor.unsqueeze(0))
        pose_enc = predictions["pose_enc"] 
        extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])

extrinsic_np = extrinsic.squeeze(0).float().cpu().numpy()

vggt_centers = []
for i in range(len(extrinsic_np)):
    mat = extrinsic_np[i]
    R_pred = mat[:3, :3]
    t_pred = mat[:3, 3]
    vggt_centers.append(-R_pred.T @ t_pred)
vggt_centers = np.array(vggt_centers)

# --- 3. ALIGNMENT & EVALUATION ---

# Split GT and VGGT into Context and Target sets
gt_context = gt_centers[:NUM_CONTEXT]
gt_targets = gt_centers[NUM_CONTEXT:]

vggt_context = vggt_centers[:NUM_CONTEXT]
vggt_targets = vggt_centers[NUM_CONTEXT:]

print(f"\n📏 Calculating Sim3 Alignment using {NUM_CONTEXT} Anchors...")

# Compute Alignment on Context Only
s, R_align, t_align = umeyama_alignment(vggt_context, gt_context)

# Apply to Context (Check fit)
context_aligned = (s * (vggt_context @ R_align.T) + t_align)
context_rmse = np.sqrt(np.mean(np.sum((context_aligned - gt_context)**2, axis=1)))

# Apply to Targets (Validation)
targets_aligned = (s * (vggt_targets @ R_align.T) + t_align)
target_errors = np.linalg.norm(targets_aligned - gt_targets, axis=1)
mean_target_error = np.mean(target_errors)

print("\n-------------------------------------------")
print(f"✅ RESULTS (Scale Factor: {s:.4f})")
print("-------------------------------------------")
print(f"🔹 Anchor Consistency (RMSE):  {context_rmse:.4f}")
print(f"🔹 Held-out Target Error (Mean): {mean_target_error:.4f}")
print("-------------------------------------------")
for i, err in enumerate(target_errors):
    print(f"   Target {i+1} ({os.path.basename(target_paths[i])}): Error = {err:.4f}")

# --- 4. SHAPE VALIDATION (Procrustes) ---
print(f"\n🔬 Checking Pure Shape Correlation (All 30 images)...")
mtx1, mtx2, disparity = procrustes(gt_centers, vggt_centers)
flat_gt = mtx1.flatten()
flat_vggt = mtx2.flatten()
correlation = np.corrcoef(flat_gt, flat_vggt)[0, 1]
print(f"Shape Correlation: {correlation:.4f}")

# --- 5. BETTER VISUALIZATION (Side-by-Side Normalized) ---
print("📊 Saving improved plot to 'trajectory_comparison.png'...")

def normalize_for_plot(points):
    """Centers and scales points to fit in a -1 to 1 box"""
    # 1. Center
    centered = points - np.mean(points, axis=0)
    # 2. Scale by max distance to keep aspect ratio
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    return centered / max_dist

# Normalize both independently for visualization
# We ignore the absolute scale difference and look purely at shape
vis_gt = normalize_for_plot(gt_centers)
vis_vggt = normalize_for_plot(vggt_centers)

# Create a figure with two subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# --- Plot 1: Ground Truth (COLMAP) ---
ax1.set_title("Ground Truth (COLMAP)\n(Normalized)", fontsize=14)
# Anchors
ax1.scatter(vis_gt[:NUM_CONTEXT, 0], vis_gt[:NUM_CONTEXT, 2], 
            c='green', marker='o', alpha=0.6, label='Anchors')
# Targets
ax1.scatter(vis_gt[NUM_CONTEXT:, 0], vis_gt[NUM_CONTEXT:, 2], 
            c='darkgreen', marker='*', s=300, label='Targets')
# Add labels for targets to track them
for i in range(NUM_TARGETS):
    idx = NUM_CONTEXT + i
    ax1.text(vis_gt[idx, 0], vis_gt[idx, 2], str(i+1), fontsize=12, fontweight='bold')

ax1.grid(True)
ax1.legend()
ax1.set_xlabel("X")
ax1.set_ylabel("Z")
ax1.axis('equal')

# --- Plot 2: Prediction (VGGT) ---
ax2.set_title(f"Prediction (VGGT)\n(Normalized) - Corr: {correlation:.3f}", fontsize=14)
# Anchors
ax2.scatter(vis_vggt[:NUM_CONTEXT, 0], vis_vggt[:NUM_CONTEXT, 2], 
            c='red', marker='x', alpha=0.6, label='Anchors')
# Targets
ax2.scatter(vis_vggt[NUM_CONTEXT:, 0], vis_vggt[NUM_CONTEXT:, 2], 
            c='darkred', marker='*', s=300, label='Targets')
# Add labels
for i in range(NUM_TARGETS):
    idx = NUM_CONTEXT + i
    ax2.text(vis_vggt[idx, 0], vis_vggt[idx, 2], str(i+1), fontsize=12, fontweight='bold')

ax2.grid(True)
ax2.legend()
ax2.set_xlabel("X")
ax2.set_ylabel("Z")
ax2.axis('equal')

plt.tight_layout()
plt.savefig("trajectory_comparison.png")
print("Done.")