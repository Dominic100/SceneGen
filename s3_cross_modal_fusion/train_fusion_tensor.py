import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MultimodalTripletDataset
from fusion_tensor_fusion import TensorFusion
import torch.optim as optim
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = MultimodalTripletDataset(r'C:\Aneesh\EDI VI\data\processed\multimodal_triplets.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TensorFusion().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()

    for epoch in range(5):
        running_loss = 0.0
        for i, (image_emb, audio_emb, text_emb) in enumerate(dataloader):
            image_emb = image_emb.to(device)
            audio_emb = audio_emb.to(device)
            text_emb = text_emb.to(device)

            optimizer.zero_grad()
            fused_emb = model(image_emb, audio_emb, text_emb)

            # Dummy target = mean projected
            target = (model.image_proj(image_emb.squeeze(1)) +
                      model.audio_proj(audio_emb.squeeze(1)) +
                      model.text_proj(text_emb.squeeze(1))) / 3.0

            loss = criterion(fused_emb, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f"Epoch [{epoch+1}/5], Step [{i+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

    # Save model
    model_path = r'C:\Aneesh\EDI VI\s3_cross_modal_fusion\fusion_tensor.pt'
    torch.save(model.state_dict(), model_path)
    print(f"✅ Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    train()
