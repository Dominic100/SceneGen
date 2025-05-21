import os
import json
from tqdm import tqdm

# === Input paths ===
TRIPLETS_PATH = r"C:\Aneesh\EDI VI\data\processed\triplets.json"
IMAGE_INDEX_PATH = r"C:\Aneesh\EDI VI\data\features\image_embeddings\image_embedding_index.json"
AUDIO_INDEX_PATH = r"C:\Aneesh\EDI VI\data\features\audio_embeddings\audio_embedding_index.json"
TEXT_INDEX_PATH = r"C:\Aneesh\EDI VI\data\features\text_embeddings\text_embedding_index.json"

# === Output path ===
OUTPUT_PATH = r"C:\Aneesh\EDI VI\data\processed\multimodal_triplets.json"

# === Load indices ===
with open(IMAGE_INDEX_PATH, "r") as f:
    image_index = json.load(f)

with open(AUDIO_INDEX_PATH, "r") as f:
    audio_index = json.load(f)

with open(TEXT_INDEX_PATH, "r") as f:
    text_index = json.load(f)

with open(TRIPLETS_PATH, "r") as f:
    triplets = json.load(f)

# === Build multimodal triplets ===
multimodal_triplets = []

for item in tqdm(triplets, desc="Building multimodal triplets"):
    # Extract IDs and paths
    image_path = item["frame_path"]
    audio_path = item["audio_path"]
    text = item["caption"]
    video_id = os.path.basename(image_path).replace(".jpg", "")

    # Match all modalities
    if image_path in image_index and audio_path in audio_index and video_id in text_index:
        multimodal_triplets.append({
            "video_id": video_id,
            "image_path": image_index[image_path],
            "audio_path": audio_index[audio_path],
            "text_path": text_index[video_id]
        })

# === Save to disk ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(multimodal_triplets, f, indent=2)

print(f"✅ Saved {len(multimodal_triplets)} aligned multimodal triplets to {OUTPUT_PATH}")
