import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.scene_generator import SceneGenerator
from data.scene_dataset import SceneDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_dataset = SceneDataset(split="train")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize model, loss, optimizer
model = SceneGenerator().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        embeddings, gt_pointcloud = batch["embedding"].to(device), batch["pointcloud"].to(device)
        optimizer.zero_grad()
        output = model(embeddings)
        loss = criterion(output, gt_pointcloud)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")