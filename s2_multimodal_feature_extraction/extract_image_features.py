# extract_image_features.py

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip

# Paths
TRIPLETS_PATH = r"C:\Aneesh\EDI VI\data\processed\triplets.json"
IMAGE_DIR = r"C:\Aneesh\EDI VI\data\frames"
OUTPUT_DIR = r"C:\Aneesh\EDI VI\data\features\image_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load triplets
with open(TRIPLETS_PATH, "r") as f:
    triplets = json.load(f)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device).eval()

# Output mapping
embedding_index = {}

for item in tqdm(triplets, desc="Extracting image features"):
    frame_path = item["frame_path"]
    video_id = os.path.basename(frame_path).replace(".jpg", "")
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}.npy")

    try:
        image = preprocess(Image.open(frame_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
            features = features / features.norm(dim=-1, keepdim=True)
        np.save(output_path, features.cpu().numpy())
        embedding_index[frame_path] = output_path
    except Exception as e:
        print(f"Failed for {frame_path}: {e}")

# Save mapping
with open(os.path.join(OUTPUT_DIR, "image_embedding_index.json"), "w") as f:
    json.dump(embedding_index, f, indent=2)
