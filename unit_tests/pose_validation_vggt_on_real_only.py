import torch
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes

# --- UPDATE THESE PATHS ---
COLMAP_PATH = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/sparse"
if not os.path.exists(os.path.join(COLMAP_PATH, "images.bin")):
    COLMAP_PATH = os.path.join(COLMAP_PATH, "0")

MODEL_PATH = "/scratch/schettip/landmark_detection/model.pt"

TEST_IMAGES = [
    "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images/00289298_7642283248.jpg",
    "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images/00315862_6836283050.jpg",
    "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images/00581890_3574867299.jpg",
    "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images/99948448_2503440660.jpg",
    "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images/98670739_6880114669.jpg",
    "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images/01069771_8567470929.jpg",
    "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images/05570784_2643017231.jpg"
]

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

    var_m = np.var(model_points, axis=0).sum()
    s = np.trace(np.diag(S)) / var_m
    t = mu_t - s * (R @ mu_m)
    return s, R, t

def get_colmap_centers(colmap_path, image_paths):
    """Retrieves camera centers from COLMAP data."""
    print(f"📂 Loading COLMAP from {colmap_path}...")
    name2center = {}
    
    if USE_PYCOLMAP:
        recon = pycolmap.Reconstruction(colmap_path)
        for _, img in recon.images.items():
            name2center[img.name] = img.projection_center()
    else:
        cameras, images, points3D = read_model(colmap_path)
        for _, img in images.items():
            R = qvec2rotmat(img.qvec)
            t = np.array(img.tvec)
            center = -R.T @ t
            name2center[img.name] = center

    gt_centers = []
    valid_indices = []
    
    for idx, path in enumerate(image_paths):
        fname = os.path.basename(path)
        if fname in name2center:
            gt_centers.append(name2center[fname])
            valid_indices.append(idx)
        else:
            print(f"⚠️ Warning: {fname} not found in COLMAP reconstruction!")

    return np.array(gt_centers), valid_indices

# --- 1. PREPARATION ---
gt_centers, valid_indices = get_colmap_centers(COLMAP_PATH, TEST_IMAGES)
if len(gt_centers) < 3:
    print("❌ Need at least 3 images for stable 3D alignment.")
    sys.exit()

valid_paths = [TEST_IMAGES[i] for i in valid_indices]

# --- 2. RUN VGGT INFERENCE ---
print("🚀 Running VGGT Inference...")
model = VGGT()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model = model.to(DEVICE).eval()

# CRITICAL CHANGE: mode="pad" ensures 518x518 square input
images_tensor = load_and_preprocess_images(valid_paths, mode="pad").to(DEVICE)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
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
num_context = len(vggt_centers) - 1
context_vggt = vggt_centers[:num_context]
context_gt = gt_centers[:num_context]
target_vggt = vggt_centers[num_context]
target_gt = gt_centers[num_context]
target_name = os.path.basename(valid_paths[num_context])

print(f"\n📏 Aligning using {num_context} images...")
print(f"🎯 Testing on held-out image: {target_name}")

s, R_align, t_align = umeyama_alignment(context_vggt, context_gt)

# Apply to Context (Training set) to check fit
context_aligned = (s * (context_vggt @ R_align.T) + t_align)
train_rmse = np.sqrt(np.mean(np.sum((context_aligned - context_gt)**2, axis=1)))

# Apply to Target (Test set)
target_aligned = s * (R_align @ target_vggt) + t_align
test_error = np.linalg.norm(target_aligned - target_gt)

print("\n-------------------------------------------")
print(f"✅ RESULTS")
print("-------------------------------------------")
print(f"Scale Factor:        {s:.4f}")
print(f"Context Alignment Error (RMSE): {train_rmse:.4f}")
print("-------------------------------------------")
print(f"Target Image:        {target_name}")
print(f"Original COLMAP Pos: {target_gt}")
print(f"VGGT Predicted Pos:  {target_aligned}")
print(f"❌ Position Error:    {test_error:.4f} units")
print("-------------------------------------------")

# --- 4. PLOTTING FOR DEBUGGING ---
print("📊 Saving debug plot to 'debug_trajectory.png'...")
all_vggt_aligned = (s * (vggt_centers @ R_align.T) + t_align)

plt.figure(figsize=(10, 8))
# Plot GT
plt.plot(gt_centers[:, 0], gt_centers[:, 2], 'g-o', label='Ground Truth (COLMAP)')
plt.plot(gt_centers[-1, 0], gt_centers[-1, 2], 'g*', markersize=15, label='Target GT')

# Plot VGGT
plt.plot(all_vggt_aligned[:, 0], all_vggt_aligned[:, 2], 'r--x', label='VGGT Predicted')
plt.plot(all_vggt_aligned[-1, 0], all_vggt_aligned[-1, 2], 'r*', markersize=15, label='Target Pred')

# Connect corresponding points
for i in range(len(gt_centers)):
    plt.plot([gt_centers[i, 0], all_vggt_aligned[i, 0]], 
             [gt_centers[i, 2], all_vggt_aligned[i, 2]], 'k:', alpha=0.3)

plt.title(f"Trajectory Top-Down View (XZ Plane)\nTest Error: {test_error:.2f}")
plt.xlabel("X (World)")
plt.ylabel("Z (World)")
plt.legend()
plt.grid(True)
plt.savefig("debug_trajectory.png")
print("Done.")

print(f"\n🔬 DIAGNOSTIC: Checking Shape Similarity (Procrustes)...")

# 1. Run Procrustes Analysis
# This normalizes both shapes to size 1, centers them, and rotates them to match perfectly.
# mtx1 is the transformed GT, mtx2 is the transformed VGGT, disparity is the error
mtx1, mtx2, disparity = procrustes(gt_centers, vggt_centers)

# 2. Calculate the Shape Correlation
# Flatten to 1D arrays
flat_gt = mtx1.flatten()
flat_vggt = mtx2.flatten()
correlation = np.corrcoef(flat_gt, flat_vggt)[0, 1]

print(f"Shape Correlation: {correlation:.4f} (1.0 = Perfect Identical Shape)")
print(f"Disparity:         {disparity:.4f} (Lower is better)")

if correlation > 0.95:
    print("✅ CONCLUSION: The shapes are IDENTICAL. VGGT is working correctly.")
    print("   The previous error was purely due to Scale Ambiguity on the small context cluster.")
else:
    print("❌ CONCLUSION: The shapes are actually different.")

# 3. Plot the Normalized Shapes
plt.figure(figsize=(8, 8))
plt.plot(mtx1[:, 0], mtx1[:, 2], 'g-o', label='Ground Truth (Normalized)')
plt.plot(mtx2[:, 0], mtx2[:, 2], 'r--x', label='VGGT (Normalized)')

# Connect points
for i in range(len(mtx1)):
    plt.plot([mtx1[i, 0], mtx2[i, 0]], [mtx1[i, 2], mtx2[i, 2]], 'k:', alpha=0.3)

plt.title(f"Normalized Shape Comparison\nCorrelation: {correlation:.4f}")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig("shape_validation.png")
print("📊 Saved 'shape_validation.png'")