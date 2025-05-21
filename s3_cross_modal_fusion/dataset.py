import json
import numpy as np
import torch
from torch.utils.data import Dataset

class MultimodalTripletDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load embeddings from npy files
        image_emb = np.load(sample['image_path'])
        audio_emb = np.load(sample['audio_path'])
        text_emb = np.load(sample['text_path'])
        
        # Convert to torch tensors (float32)
        image_emb = torch.tensor(image_emb, dtype=torch.float32)
        audio_emb = torch.tensor(audio_emb, dtype=torch.float32)
        text_emb = torch.tensor(text_emb, dtype=torch.float32)
        
        return image_emb, audio_emb, text_emb


if __name__ == "__main__":
    # Quick test
    dataset = MultimodalTripletDataset(r'C:\Aneesh\EDI VI\data\processed\multimodal_triplets.json')
    print(f"Dataset size: {len(dataset)}")
    
    image_emb, audio_emb, text_emb = dataset[0]
    print(f"Image embedding shape: {image_emb.shape}")
    print(f"Audio embedding shape: {audio_emb.shape}")
    print(f"Text embedding shape: {text_emb.shape}")
