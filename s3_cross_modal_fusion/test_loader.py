import torch
from torch.utils.data import DataLoader
from dataset import MultimodalTripletDataset

def main():
    dataset = MultimodalTripletDataset(r'C:\Aneesh\EDI VI\data\processed\multimodal_triplets.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for batch_idx, (image_emb, audio_emb, text_emb) in enumerate(dataloader):
        print(f"Batch {batch_idx} - image: {image_emb.shape}, audio: {audio_emb.shape}, text: {text_emb.shape}")
        break  # Just check one batch

if __name__ == "__main__":
    main()
