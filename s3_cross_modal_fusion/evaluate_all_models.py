import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MultimodalTripletDataset
from fusion_concat_mlp import ConcatMLPFusion
from fusion_attention import AttentionFusion
from fusion_cross_attention import CrossAttentionFusion
from fusion_tensor_fusion import TensorFusion

def load_model(model_class, input_dims, model_path, device):
    # Check if model expects unpacked arguments
    if model_class in [AttentionFusion, CrossAttentionFusion, TensorFusion]:
        # Pass dimensions as separate arguments with an output_dim
        model = model_class(*input_dims, output_dim=512).to(device)
    else:
        # ConcatMLPFusion expects input_dims as a single argument
        model = model_class(input_dims).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, dataloader, device, criterion):
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for image_emb, audio_emb, text_emb in dataloader:
            image_emb = image_emb.to(device)
            audio_emb = audio_emb.to(device)
            text_emb = text_emb.to(device)

            # Get the dimensions
            image = image_emb.squeeze(1)  # (B, 512)
            audio = audio_emb.squeeze(1)  # (B, 2048)
            text = text_emb.squeeze(1)    # (B, 512)

            # Project audio to match dimensions (just like in training)
            audio_proj = nn.Linear(audio.shape[1], image.shape[1]).to(device)
            audio_mapped = audio_proj(audio)  # Project to 512

            # Target: average of the three modalities with matching dimensions
            target = (image + audio_mapped + text) / 3.0

            output = model(image_emb, audio_emb, text_emb)

            loss = criterion(output, target)
            total_loss += loss.item() * image_emb.size(0)
            count += image_emb.size(0)

    return total_loss / count

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = MultimodalTripletDataset(r'C:\Aneesh\EDI VI\data\processed\multimodal_triplets.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Extract input dims (image_dim, audio_dim, text_dim)
    img_emb, audio_emb, text_emb = dataset[0]
    input_dims = (img_emb.shape[-1], audio_emb.shape[-1], text_emb.shape[-1])

    criterion = torch.nn.MSELoss()

    model_paths = {
        'Concat + MLP': r'C:\Aneesh\EDI VI\s3_cross_modal_fusion\fusion_concat_mlp.pt',
        'Attention Fusion': r'C:\Aneesh\EDI VI\s3_cross_modal_fusion\fusion_attention.pt',
        'Cross Attention Fusion': r'C:\Aneesh\EDI VI\s3_cross_modal_fusion\fusion_cross_attention.pt',
        'Tensor Fusion': r'C:\Aneesh\EDI VI\s3_cross_modal_fusion\fusion_tensor.pt'
    }

    model_classes = {
        'Concat + MLP': ConcatMLPFusion,
        'Attention Fusion': AttentionFusion,
        'Cross Attention Fusion': CrossAttentionFusion,
        'Tensor Fusion': TensorFusion
    }

    results = {}
    for name in model_paths:
        print(f"Evaluating {name}...")
        model = load_model(model_classes[name], input_dims, model_paths[name], device)
        loss = evaluate_model(model, dataloader, device, criterion)
        results[name] = loss
        print(f"{name} MSE Loss: {loss:.6f}")

    print("\n=== Evaluation Summary ===")
    for name, loss in results.items():
        print(f"{name}: MSE Loss = {loss:.6f}")

if __name__ == "__main__":
    main()