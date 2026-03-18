# AI Landmark Detection & Geometric Forensics

## Introduction
This repository contains a pipeline for analyzing and authenticating images of famous landmarks (e.g., Brandenburg Gate, Taj Mahal). Given an AI-generated or unverified image, the system estimates its 3D camera pose relative to a real dataset, retrieves or renders a real-world counterpart from that exact viewpoint, and performs rigid geometric forensics using feature matching to detect anomalies.

---

## Table of Contents
- [Core Pipeline](#core-pipeline)
- [Project Structure](#project-structure)
- [Installation & Dependencies](#installation--dependencies)
- [Usage Guide](#usage-guide)
- [Methodology](#methodology)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

---

## Core Pipeline

### Camera Pose Estimation
Uses VGGT (Visual Geometry Grounding Transformer) to predict extrinsic and intrinsic camera parameters of a target image by comparing it to multiple real anchor images from a COLMAP reconstruction.

### Space Alignment
Aligns the VGGT predicted coordinate system with the real-world coordinate system using Sim3 (Umeyama) alignment.

### Counterpart Generation

#### Retrieval
`find_closest_real_image_final.py` finds the closest real image using:
- 3D distance
- Rotation similarity
- Field-of-view (FOV)
- Visual reranking with LightGlue

#### 3D Rendering
`render_ai_pose_2.py` renders a real-world equivalent view using 3D Gaussian Splatting (gsplat) from the estimated camera pose.

### Interactive Forensics
`interactive_lightglue_inverse.py` provides a Gradio interface that:
- Extracts features using SuperPoint
- Matches features using LightGlue
- Allows cropping regions to measure pose drift
- Detects inconsistencies between foreground and background geometry

---

## Project Structure

### Pose Extraction & Alignment
- `run_ai_pose_extraction.py`, `run_vggt_on_AI_v1.py`: Run VGGT inference against COLMAP anchors and output aligned pose
- `colmap_io.py`: Custom loader for COLMAP `.bin` files (cameras, images, points3D)

### Counterpart Generation
- `find_closest_real_image_final.py`: Retrieval pipeline with geometric filtering and visual reranking
- `render_ai_pose_2.py`, `final_render_vggt.py`: Render novel views using 3D Gaussian Splatting

### Interactive Geometric Forensics (UIs)
- `interactive_lightglue_inverse.py`: Crop-based feature matching and pose drift visualization
- `interactive_vggt.py`, `interactive_vggt_simpler.py`: Compare VGGT base drift vs masked drift

### Unit Tests (`unit_tests/`)
- `pose_validation_vggt_on_real_batch.py`, `pose_validation_vggt_on_real_only.py`: Validate Sim3 alignment and Procrustes correlation
- `debug_pose_math.py`: Validate rotation matrix conversions
- `validate_focal_length_pred.py`: Validate focal length predictions

---

## Installation & Dependencies

Ensure a CUDA-capable GPU is available. The project uses `torch.bfloat16` optimizations.

### Core Dependencies
```bash
pip install torch torchvision numpy scipy pillow matplotlib gradio imageio

Feature Matching
pip install lightglue
3D Rendering & Geometry
pip install plyfile
pip install gsplat
Optional
pip install pycolmap

Additional requirements:

VGGT model file (model.pt)

Local COLMAP datasets

3D Gaussian Splatting .ply files

Usage Guide
1. Retrieve Closest Real Image
python find_closest_real_image_final.py \
    --target_img "/path/to/ai_generated_image.jpeg" \
    --output "retrieved_match.jpg"
2. Render AI Pose Using 3DGS
python render_ai_pose_2.py \
    --target_img "/path/to/ai_generated_image.jpeg" \
    --ply_path "/path/to/3dgs_point_cloud.ply" \
    --colmap_path "/path/to/colmap/sparse" \
    --image_dir "/path/to/colmap/images" \
    --model_path "/path/to/vggt_model.pt" \
    --output "final_render.png" \
    --lift_amount 0.2
3. Run Interactive Forensics UI
python interactive_lightglue_inverse.py
UI Instructions

Upload a real reference image (anchor)

Upload the target image

Select a region using two clicks (bounding box)

Click "Analyze Rigid Geometry"

Inspect feature matches and pose drift results

Methodology
Anchor Selection

Selects approximately 50 random real images from a COLMAP dataset to create a stable anchor set.

Sim3 Alignment

VGGT outputs poses in an arbitrary coordinate system. Camera centers are aligned to the COLMAP world space using Umeyama alignment to compute:

Scale (s)

Rotation (R)

Translation (t)

FOV-Based Scoring

Retrieval scoring combines:

LightGlue match count

Penalty based on FOV difference

Pose Drift Detection

Rigid structures should maintain consistent geometry. If an AI-generated image mixes inconsistent foreground and background elements:

Cropping foreground vs background leads to different pose estimations

High pose drift indicates geometric inconsistency

Troubleshooting

Ensure CUDA is properly configured for PyTorch

Verify paths to COLMAP datasets and .ply files

Confirm VGGT model file is correctly loaded

If rendering fails, check gsplat installation and compatibility

For poor matches, ensure sufficient overlap between target and dataset images