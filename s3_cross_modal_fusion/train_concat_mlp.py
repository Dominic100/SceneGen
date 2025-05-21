import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MultimodalTripletDataset
from fusion_concat_mlp import ConcatMLPFusion
import torch.optim as optim

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = MultimodalTripletDataset(r'C:\Aneesh\EDI VI\data\processed\multimodal_triplets.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Get input dimensions from one sample
    img_emb, audio_emb, text_emb = dataset[0]
    input_dims = (img_emb.shape[-1], audio_emb.shape[-1], text_emb.shape[-1])  # (512, 2048, 512)

    # Initialize model
    model = ConcatMLPFusion(input_dims=input_dims, output_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()

    for epoch in range(5):  # Light training for prototype
        running_loss = 0.0
        for i, (image_emb, audio_emb, text_emb) in enumerate(dataloader):
            image_emb = image_emb.to(device)  # (B, 1, 512)
            audio_emb = audio_emb.to(device)  # (B, 1, 2048)
            text_emb = text_emb.to(device)    # (B, 1, 512)

            optimizer.zero_grad()

            # Forward pass
            fused_emb = model(image_emb, audio_emb, text_emb)  # (B, 512)

            # Use average of inputs as dummy target
            image = image_emb.squeeze(1)  # (B, 512)
            audio = audio_emb.squeeze(1)  # (B, 2048)
            text = text_emb.squeeze(1)    # (B, 512)

            # Project audio to 512 to match others for averaging (optional dummy target projection)
            audio_proj = nn.Linear(audio.shape[1], 512).to(device)
            audio_mapped = audio_proj(audio)  # (B, 512)

            target = (image + audio_mapped + text) / 3.0  # (B, 512)

            loss = criterion(fused_emb, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f"Epoch [{epoch+1}/5], Step [{i+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

    print("✅ Training complete for Concat + MLP Fusion.")

    torch.save(model.state_dict(), 'concat_mlp_fusion.pth')
    print("✅ Model saved to concat_mlp_fusion.pth")


if __name__ == "__main__":
    train()
