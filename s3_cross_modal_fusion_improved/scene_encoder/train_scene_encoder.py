import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from scene_transformer_encoder import SceneTransformerEncoder

# ===================== Dataset =====================

class TemporalTripletDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def load_modalities(self, entries):
        image_feats, audio_feats, text_feats = [], [], []
        for entry in entries:
            img = np.load(entry['image_path'])
            audio = np.load(entry['audio_path'])
            text = np.load(entry['text_path'])
            
            image_feats.append(img)
            audio_feats.append(audio)
            text_feats.append(text)
        
        # Stack features along the sequence dimension
        img_stack = np.stack(image_feats)
        audio_stack = np.stack(audio_feats)
        text_stack = np.stack(text_feats)
        
        img_tensor = torch.tensor(img_stack, dtype=torch.float32)
        audio_tensor = torch.tensor(audio_stack, dtype=torch.float32)
        text_tensor = torch.tensor(text_stack, dtype=torch.float32)
        
        return (img_tensor, audio_tensor, text_tensor)

    def __getitem__(self, idx):
        sample = self.data[idx]
        anchor = self.load_modalities(sample['anchor']['entries'])
        positive = self.load_modalities(sample['positive']['entries'])
        negative = self.load_modalities(sample['negative']['entries'])
        return anchor, positive, negative

# ===================== Loss =====================

def triplet_loss(anchor, positive, negative, margin=0.2):
    distance_pos = torch.norm(anchor - positive, dim=1)
    distance_neg = torch.norm(anchor - negative, dim=1)
    loss = torch.clamp(distance_pos - distance_neg + margin, min=0.0)
    return loss.mean()

# ===================== Training =====================

def train(model, dataloader, optimizer, device, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (anchor, positive, negative) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            anchor_feats = [x.to(device) for x in anchor]
            positive_feats = [x.to(device) for x in positive]
            negative_feats = [x.to(device) for x in negative]
            
            anchor_out = model(*anchor_feats)
            positive_out = model(*positive_feats)
            negative_out = model(*negative_feats)
            
            loss = triplet_loss(anchor_out, positive_out, negative_out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

# ===================== Main =====================

if __name__ == "__main__":
    json_path = r"C:\Aneesh\EDI VI\data\processed\scene_triplets.json"
    batch_size = 8
    epochs = 10
    lr = 1e-4

    dataset = TemporalTripletDataset(json_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SceneTransformerEncoder(
        img_emb_dim=512,
        audio_emb_dim=2048,
        text_emb_dim=512,
        fusion_emb_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_seq_len=5,
        output_dim=512
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train(model, dataloader, optimizer, device, epochs=epochs)
    
    os.makedirs(r"C:\Aneesh\EDI VI\s3_cross_modal_fusion_improved\models", exist_ok=True)
    model_path = r"C:\Aneesh\EDI VI\s3_cross_modal_fusion_improved\models\scene_encoder_transformer.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")