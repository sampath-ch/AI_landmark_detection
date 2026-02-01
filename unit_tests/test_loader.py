import os
import sys


# UPDATE THIS PATH to where your 'sparse/0' or 'sparse' folder is
# It should be the folder containing 'images.bin', 'cameras.bin', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from colmap_io import read_model
colmap_path = "/scratch/schettip/landmark_detection/datasets/brandeburg_gate_data/brandenburg_gate/dense/sparse" 

print(f"Loading model from: {colmap_path}")

try:
    cameras, images, points3D = read_model(colmap_path)
    
    print("---------------------------------------")
    print(f"✅ Successfully loaded COLMAP data!")
    print(f"📸 Number of Cameras: {len(cameras)}")
    print(f"🖼️  Number of Images: {len(images)}")
    print(f"☁️  Number of 3D Points: {len(points3D)}")
    
    # Print one example image name to be sure
    first_image_id = list(images.keys())[0]
    print(f"Example Image Name: {images[first_image_id].name}")
    print("---------------------------------------")

except FileNotFoundError:
    print("❌ Error: Could not find .bin files. Check your path.")
except Exception as e:
    print(f"❌ Error: {e}")