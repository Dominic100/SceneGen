# train_fusion_attention.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MultimodalTripletDataset
from fusion_attention import AttentionFusion
import torch.optim as optim

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = MultimodalTripletDataset(r'C:\Aneesh\EDI VI\data\processed\multimodal_triplets.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Get input dimensions from one sample
    img_emb, audio_emb, text_emb = dataset[0]
    input_dims = (img_emb.shape[-1], audio_emb.shape[-1], text_emb.shape[-1])  # (512, 2048, 512)

    # Initialize model
    model = AttentionFusion(*input_dims, output_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()

    for epoch in range(5):  # Light training
        running_loss = 0.0
        for i, (image_emb, audio_emb, text_emb) in enumerate(dataloader):
            image_emb = image_emb.to(device)
            audio_emb = audio_emb.to(device)
            text_emb = text_emb.to(device)

            optimizer.zero_grad()

            # Forward pass
            fused_emb = model(image_emb, audio_emb, text_emb)

            # Project modalities to same space for target
            target = model.project_modalities(image_emb, audio_emb, text_emb)

            loss = criterion(fused_emb, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f"Epoch [{epoch+1}/5], Step [{i+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

    print("✅ Training complete for Attention-Based Fusion.")

    # Save model
    torch.save(model.state_dict(), "fusion_attention.pt")
    print("✅ Model saved to fusion_attention.pt.")

if __name__ == "__main__":
    train()
