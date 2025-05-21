# extract_audio_features.py

import os
import json
import torch
import librosa
import numpy as np
from tqdm import tqdm

from torchlibrosa.stft import LogmelFilterBank
from torch import nn

# PANNs CNN14 definition (from official repo)
class PANNsCNN14(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        self.model = checkpoint['model']
    
    def forward(self, waveform):
        self.model.eval()
        with torch.no_grad():
            output_dict = self.model(waveform)
            return output_dict['embedding']

# Paths
TRIPLET_PATH = r"C:\Aneesh\EDI VI\data\processed\triplets.json"
AUDIO_DIR = r"C:\Aneesh\EDI VI\data\audio"
OUTPUT_DIR = r"C:\Aneesh\EDI VI\data\features\audio_embeddings"
PRETRAINED_PATH = r"C:\Aneesh\EDI VI\data\pretrained\Cnn14_16k.pth"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load triplets
with open(TRIPLET_PATH, "r") as f:
    triplets = json.load(f)

# Device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(PRETRAINED_PATH, map_location=device)
model = checkpoint['model']
model.to(device)
model.eval()

# Feature extraction
embedding_index = {}

for item in tqdm(triplets, desc="Extracting audio features"):
    audio_path = item["audio_path"]
    audio_id = os.path.basename(audio_path).replace(".wav", "")
    out_path = os.path.join(OUTPUT_DIR, f"{audio_id}.npy")

    try:
        waveform, sr = librosa.load(audio_path, sr=16000)
        waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(device)

        with torch.no_grad():
            output_dict = model(waveform_tensor)
            embedding = output_dict["embedding"].cpu().numpy()

        np.save(out_path, embedding)
        embedding_index[audio_path] = out_path

    except Exception as e:
        print(f"[ERROR] {audio_path}: {e}")

# Save mapping
with open(os.path.join(OUTPUT_DIR, "audio_embedding_index.json"), "w") as f:
    json.dump(embedding_index, f, indent=2)
