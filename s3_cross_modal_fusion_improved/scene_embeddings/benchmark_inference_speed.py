# benchmark_inference_speed.py

import os
import json
import time
import torch
import pandas as pd
import numpy as np
import sys

sys.path.append(r"C:\Aneesh\EDI VI\s3_cross_modal_fusion_improved\scene_encoder")
from scene_transformer_encoder import SceneTransformerEncoder

# CONFIG
TRIPLETS_PATH = r"C:\Aneesh\EDI VI\data\processed\scene_triplets.json"
MODEL_PATH = r"C:\Aneesh\EDI VI\s3_cross_modal_fusion_improved\models\scene_encoder_transformer.pt"
OUTPUT_CSV = "inference_times.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model dims
IMG_DIM = 512
AUDIO_DIM = 2048
TEXT_DIM = 512
MAX_SEQ_LEN = 5

# Load model
model = SceneTransformerEncoder(
    img_emb_dim=IMG_DIM,
    audio_emb_dim=AUDIO_DIM,
    text_emb_dim=TEXT_DIM,
    max_seq_len=MAX_SEQ_LEN,
    output_dim=512,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load data
with open(TRIPLETS_PATH, "r") as f:
    triplets = json.load(f)

# Unique scenes
scene_id_map = {}
for t in triplets:
    for key in ["anchor", "positive", "negative"]:
        scene = t[key]
        scene_id = scene["scene_id"]
        if scene_id not in scene_id_map:
            scene_id_map[scene_id] = scene

# Benchmarking
results = []
for scene_id, scene in scene_id_map.items():
    # Load and stack feature arrays
    img_feats = []
    audio_feats = []
    text_feats = []
    for entry in scene["entries"]:
        img_feats.append(np.load(entry["image_path"]))
        audio_feats.append(np.load(entry["audio_path"]))
        text_feats.append(np.load(entry["text_path"]))
    img_tensor = torch.tensor(np.stack(img_feats)).unsqueeze(0).to(DEVICE)
    audio_tensor = torch.tensor(np.stack(audio_feats)).unsqueeze(0).to(DEVICE)
    text_tensor = torch.tensor(np.stack(text_feats)).unsqueeze(0).to(DEVICE)

    # Time inference
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    start = time.time()
    with torch.no_grad():
        _ = model(img_tensor, audio_tensor, text_tensor)
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    end = time.time()

    inference_time_ms = (end - start) * 1000
    video_id = scene["video_id"]
    results.append({"scene_id": scene_id, "video_id": video_id, "inference_time_ms": round(inference_time_ms, 3)})

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved inference times to {OUTPUT_CSV}")
