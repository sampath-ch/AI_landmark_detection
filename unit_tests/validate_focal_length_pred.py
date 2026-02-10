import numpy as np
import torch
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from PIL import Image
import sys
import os

# CONFIG
MODEL_PATH = "/scratch/schettip/landmark_detection/model.pt"
TARGET_IMG = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/images/05570784_2643017231.jpg"
COLMAP_PATH = "/scratch/schettip/landmark_detection/datasets/brandenburg_gate/dense/sparse"
if not os.path.exists(os.path.join(COLMAP_PATH, "images.bin")): COLMAP_PATH = os.path.join(COLMAP_PATH, "0")

# HELPERS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
try: import pycolmap; USE_PYCOLMAP=True
except: from colmap_io import read_model; USE_PYCOLMAP=False

def get_gt_focal(colmap_path, img_name):
    if USE_PYCOLMAP:
        recon = pycolmap.Reconstruction(colmap_path)
        for _, img in recon.images.items():
            if img.name == img_name: return recon.cameras[img.camera_id].params[0]
    else:
        cameras, images, _ = read_model(colmap_path)
        for _, img in images.items():
            if img.name == img_name: return cameras[img.camera_id].params[0]
    return None

# MAIN
model = VGGT()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model = model.to("cuda").eval()

# 1. Get Prediction
img_tensor = load_and_preprocess_images([TARGET_IMG], mode="pad").to("cuda")
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    pred = model(img_tensor.unsqueeze(0))
    _, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], (518, 518))

focal_pred_518 = intrinsic[0, 0, 0, 0].item()

# 2. Scale to Original
with Image.open(TARGET_IMG) as img: w, h = img.size
scale = 518.0 / max(w, h)
focal_pred_orig = focal_pred_518 / scale

# 3. Compare
focal_gt = get_gt_focal(COLMAP_PATH, os.path.basename(TARGET_IMG))

print(f"--- FOCAL LENGTH CHECK ---")
print(f"GT Focal Length:   {focal_gt:.2f}")
print(f"Pred Focal Length: {focal_pred_orig:.2f}")
print(f"Error:             {abs(focal_gt - focal_pred_orig):.2f} px")
print(f"Error %:           {abs(focal_gt - focal_pred_orig)/focal_gt*100:.2f}%")