# generate_scene_embeddings.py

import os
import json
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import sys
import numpy as np

# Add path to scene encoder module
sys.path.append(r"C:\Aneesh\EDI VI\s3_cross_modal_fusion_improved\scene_encoder")

from scene_transformer_encoder import SceneTransformerEncoder  # Ensure this is correct

# CONFIG
TRIPLETS_PATH = r"C:\Aneesh\EDI VI\data\processed\scene_triplets.json"
MODEL_PATH = r"C:\Aneesh\EDI VI\s3_cross_modal_fusion_improved\models\scene_encoder_transformer.pt"
OUTPUT_CSV_PATH = "scene_embeddings.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model input dimensions
IMG_DIM = 512
AUDIO_DIM = 2048
TEXT_DIM = 512
MAX_SEQ_LEN = 5
OUTPUT_DIM = 512

# Load scene encoder model
model = SceneTransformerEncoder(
    img_emb_dim=IMG_DIM,
    audio_emb_dim=AUDIO_DIM,
    text_emb_dim=TEXT_DIM,
    max_seq_len=MAX_SEQ_LEN,
    output_dim=OUTPUT_DIM,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Helper: Load and concatenate image/audio/text feature tensors
def load_scene_embedding(scene):
    # Load and stack features
    img_features = []
    audio_features = []
    text_features = []

    for entry in scene["entries"]:
        img_features.append(np.load(entry["image_path"]))
        audio_features.append(np.load(entry["audio_path"]))
        text_features.append(np.load(entry["text_path"]))

    img_tensor = torch.tensor(np.stack(img_features)).unsqueeze(0).to(DEVICE)       # (1, T, 512)
    audio_tensor = torch.tensor(np.stack(audio_features)).unsqueeze(0).to(DEVICE)   # (1, T, 2048)
    text_tensor = torch.tensor(np.stack(text_features)).unsqueeze(0).to(DEVICE)     # (1, T, 512)

    with torch.no_grad():
        scene_embedding = model(img_tensor, audio_tensor, text_tensor).squeeze(0).cpu().numpy()
    return scene_embedding

# Load triplets
with open(TRIPLETS_PATH, "r") as f:
    triplets = json.load(f)

# Collect unique scenes using scene_id
scene_id_map = {}  # scene_id -> scene_dict
for triplet in triplets:
    for key in ["anchor", "positive", "negative"]:
        scene = triplet[key]
        scene_id = scene["scene_id"]
        if scene_id not in scene_id_map:
            scene_id_map[scene_id] = scene

print(f"Found {len(scene_id_map)} unique scenes to embed.")

# Generate and collect embeddings
embeddings = []
for scene_id, scene_data in tqdm(scene_id_map.items(), desc="Generating scene embeddings"):
    embedding = load_scene_embedding(scene_data)
    embeddings.append({
        "scene_id": scene_id,
        **{f"dim_{i}": float(val) for i, val in enumerate(embedding)}
    })

# Save to CSV
df = pd.DataFrame(embeddings)
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Saved scene embeddings to {OUTPUT_CSV_PATH}")
