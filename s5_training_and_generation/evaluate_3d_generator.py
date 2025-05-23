import torch
from torch.utils.data import DataLoader
from models.scene_generator import SceneGenerator
from data.scene_dataset import SceneDataset
from utils.metrics import chamfer_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SceneGenerator().to(device)
model.load_state_dict(torch.load("checkpoints/model_epoch_50.pt"))
model.eval()

# Load dataset
val_dataset = SceneDataset(split="val")
val_loader = DataLoader(val_dataset, batch_size=1)

# Evaluate
total_cd = 0
for batch in val_loader:
    embedding = batch["embedding"].to(device)
    gt_pointcloud = batch["pointcloud"].to(device)
    with torch.no_grad():
        pred_pointcloud = model(embedding)
    cd = chamfer_distance(pred_pointcloud, gt_pointcloud)
    total_cd += cd.item()

print(f"Average Chamfer Distance on validation set: {total_cd / len(val_loader):.6f}")
