import os
import json
import torch
import numpy as np
from tqdm import tqdm
import open_clip

TRIPLET_PATH = r"C:\Aneesh\EDI VI\data\processed\triplets.json"
OUTPUT_DIR = r"C:\Aneesh\EDI VI\data\features\text_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(TRIPLET_PATH, "r") as f:
    triplets = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device).eval()

embedding_index = {}
failed_captions = []

for i, item in enumerate(tqdm(triplets, desc="Extracting text features")):
    caption = item["caption"].strip()  # trim whitespace
    frame_path = item["frame_path"]
    video_id = os.path.splitext(os.path.basename(frame_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}.npy")

    if not caption:
        print(f"[WARN] Empty caption for {video_id}, skipping.")
        failed_captions.append(video_id)
        continue

    try:
        text = open_clip.tokenize([caption]).to(device)
        with torch.no_grad():
            features = model.encode_text(text)
            features = features / features.norm(dim=-1, keepdim=True)
        np.save(output_path, features.cpu().numpy())
        # Use video_id as key (unique)
        embedding_index[video_id] = output_path

    except Exception as e:
        print(f"[ERROR] Failed for {video_id} ({caption}): {e}")
        failed_captions.append(video_id)

    if i > 0 and i % 500 == 0:
        print(f"Processed {i} captions...")

print(f"Total failed captions: {len(failed_captions)}")
if failed_captions:
    print("Failed video IDs:", failed_captions)

with open(os.path.join(OUTPUT_DIR, "text_embedding_index.json"), "w") as f:
    json.dump(embedding_index, f, indent=2)
