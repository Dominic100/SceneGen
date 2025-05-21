# extract_audio_features.py

import os
import json
import numpy as np
import librosa
from tqdm import tqdm

from panns_inference import AudioTagging

# === Paths ===
TRIPLET_PATH = r"C:\Aneesh\EDI VI\data\processed\triplets.json"
AUDIO_DIR = r"C:\Aneesh\EDI VI\data\audio"
OUTPUT_DIR = r"C:\Aneesh\EDI VI\data\features\audio_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load triplets ===
with open(TRIPLET_PATH, "r") as f:
    triplets = json.load(f)

# === Initialize model ===
at = AudioTagging(checkpoint_path=None, device='cuda')  # use 'cpu' if needed

# === Feature extraction ===
embedding_index = {}

for item in tqdm(triplets, desc="Extracting audio embeddings"):
    audio_path = item["audio_path"]
    audio_id = os.path.basename(audio_path).replace(".wav", "")
    out_path = os.path.join(OUTPUT_DIR, f"{audio_id}.npy")

    try:
        waveform, sr = librosa.load(audio_path, sr=32000, mono=True)
        audio_tensor = waveform[None, :]  # shape: (1, num_samples)

        _, embedding = at.inference(audio_tensor)
        np.save(out_path, embedding)
        embedding_index[audio_path] = out_path

    except Exception as e:
        print(f"[ERROR] {audio_path}: {e}")

# === Save mapping ===
with open(os.path.join(OUTPUT_DIR, "audio_embedding_index.json"), "w") as f:
    json.dump(embedding_index, f, indent=2)
