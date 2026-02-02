import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from colmap_io import read_model

# --- NEW IMPORT ---
try:
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except ImportError:
    print("❌ Error: Could not import VGGT utils.")
    exit()

# --- CONFIGURATION ---
COLMAP_PATH = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/sparse"
IMAGES_ROOT = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images"
AI_IMAGE_PATH = "/scratch/schettip/landmark_detection/ai_generated.jpeg"
# !!! UPDATE THIS TO YOUR MODEL PATH !!!
LOCAL_MODEL_PATH = "/scratch/schettip/landmark_detection/model.pt" 

NUM_REF_IMAGES = 50 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HELPER: QUATERNION TO ROTATION MATRIX ---
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

# --- HELPER: UMEYAMA ALIGNMENT (VGGT -> COLMAP) ---
def align_sim3(model_points, target_points):
    """Computes Scale, Rotation, and Translation to align model_points to target_points."""
    mu_m = model_points.mean(0)
    mu_t = target_points.mean(0)
    m_centered = model_points - mu_m
    t_centered = target_points - mu_t
    
    # Scale
    sigma2 = (m_centered**2).sum() / len(model_points)
    correlation = (t_centered.T @ m_centered) / len(model_points)
    
    U, D, Vt = np.linalg.svd(correlation)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sigma2
    t = mu_t - s * (R @ mu_m)
    
    return s, R, t

# --- 1. Load Data ---
print(f"📂 Loading COLMAP model...")
cameras, images, points3D = read_model(COLMAP_PATH)

# --- 2. Select Reference Images & Get Ground Truth ---
print(f"🎲 Selecting {NUM_REF_IMAGES} reference images...")
all_image_ids = list(images.keys())
selected_ids = random.sample(all_image_ids, NUM_REF_IMAGES)

ref_paths = []
gt_centers = [] # Ground Truth Camera Centers (XYZ)

for img_id in selected_ids:
    img_data = images[img_id]
    ref_paths.append(os.path.join(IMAGES_ROOT, img_data.name))
    
    # FIX: Use the standalone function instead of the object method
    R_gt = qvec2rotmat(img_data.qvec) 
    t_gt = np.array(img_data.tvec)
    center = -R_gt.T @ t_gt
    gt_centers.append(center)

input_paths = ref_paths + [AI_IMAGE_PATH]
gt_centers = np.array(gt_centers) # Shape (N, 3)

# --- 3. Run Inference ---
print("🚀 Loading Model & Running Inference...")
from vggt.models.vggt import VGGT
model = VGGT()
state_dict = torch.load(LOCAL_MODEL_PATH, map_location="cpu") 
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

# Preprocess
batch_tensors = []
for p in input_paths:
    img = Image.open(p).convert("RGB")
    # Resize logic (max 518, div 14)
    w, h = img.size
    scale = 518 / max(w, h)
    new_w, new_h = (int(w*scale)//14)*14, (int(h*scale)//14)*14
    img = img.resize((new_w, new_h), Image.BICUBIC)
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img).to(DEVICE)
    batch_tensors.append(t)

# Pad and Stack
max_h, max_w = max(t.shape[1] for t in batch_tensors), max(t.shape[2] for t in batch_tensors)
padded = [F.pad(t, (0, max_w-t.shape[2], 0, max_h-t.shape[1])) for t in batch_tensors]
batch_input = torch.stack(padded).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    agg_tokens, ps_idx = model.aggregator(batch_input)
    pose_enc = model.camera_head(agg_tokens)[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch_input.shape[-2:])

# --- 4. Validation & Alignment ---
# Convert VGGT extrinsics to Camera Centers
pred_matrices = extrinsic.squeeze(0).cpu().numpy() # (N+1, 3, 4) or (N+1, 4, 4)
vggt_centers = []

for i in range(NUM_REF_IMAGES): # Only the real images
    mat = pred_matrices[i] # 3x4 or 4x4
    R_pred = mat[:3, :3]
    t_pred = mat[:3, 3]
    # Compute Center: C = -R^T * t
    c_pred = -R_pred.T @ t_pred
    vggt_centers.append(c_pred)

vggt_centers = np.array(vggt_centers)

# Align VGGT Centers -> COLMAP Centers
s, R_align, t_align = align_sim3(vggt_centers, gt_centers)

# Calculate Error
aligned_vggt = s * (vggt_centers @ R_align.T) + t_align
error = np.linalg.norm(aligned_vggt - gt_centers, axis=1).mean()

print(f"\n📊 ALIGNMENT ERROR (Real Images): {error:.4f} COLMAP units")
if error < 1.5: 
    print("✅ alignment is GOOD. You can trust the AI pose.")
else:
    print("⚠️ alignment is POOR. The AI pose might be inaccurate.")

# --- 5. Transform AI Image to COLMAP Space ---
ai_mat = pred_matrices[-1]
ai_R = ai_mat[:3, :3]
ai_t = ai_mat[:3, 3]
ai_center_vggt = -ai_R.T @ ai_t

# Apply Sim3 to the AI Center
ai_center_colmap = s * (R_align @ ai_center_vggt) + t_align

print("\n📍 FINAL AI POSITION (COLMAP COORDINATES):")
print(f"X: {ai_center_colmap[0]:.4f}")
print(f"Y: {ai_center_colmap[1]:.4f}")
print(f"Z: {ai_center_colmap[2]:.4f}")