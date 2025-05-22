import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from dataset import MultimodalTripletDataset
from fusion_concat_mlp import ConcatMLPFusion
from fusion_attention import AttentionFusion
from fusion_cross_attention import CrossAttentionFusion
from fusion_tensor_fusion import TensorFusion

def load_model(model_class, input_dims, model_path, device):
    if model_class in [AttentionFusion, CrossAttentionFusion, TensorFusion]:
        model = model_class(*input_dims, output_dim=512).to(device)
    else:
        model = model_class(input_dims).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def qualitative_sample(model, dataloader, device, model_name, audio_proj):
    rows = []
    with torch.no_grad():
        for i, (image_emb, audio_emb, text_emb) in enumerate(dataloader):
            if i >= 5:  # Take only 5 batches
                break

            image = image_emb.squeeze(1).to(device)
            audio = audio_emb.squeeze(1).to(device)
            text = text_emb.squeeze(1).to(device)

            audio_mapped = audio_proj(audio)
            target = (image + audio_mapped + text) / 3.0

            output = model(image_emb.to(device), audio_emb.to(device), text_emb.to(device))

            for j in range(min(3, output.size(0))):  # Take only 3 samples per batch
                rows.append({
                    "Model": model_name,
                    "SampleIndex": i * 32 + j,
                    "PredictedEmbedding": output[j].cpu().numpy().tolist(),
                    "TargetEmbedding": target[j].cpu().numpy().tolist()
                })
    return rows

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MultimodalTripletDataset(r'C:\Aneesh\EDI VI\data\processed\multimodal_triplets.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    img_emb, audio_emb, text_emb = dataset[0]
    input_dims = (img_emb.shape[-1], audio_emb.shape[-1], text_emb.shape[-1])
    audio_proj = nn.Linear(input_dims[1], input_dims[0]).to(device)

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

    all_rows = []
    for name in model_paths:
        print(f"Running qualitative evaluation for {name}...")
        model = load_model(model_classes[name], input_dims, model_paths[name], device)
        rows = qualitative_sample(model, dataloader, device, name, audio_proj)
        all_rows.extend(rows)

    # Convert to DataFrame and save
    df = pd.DataFrame(all_rows)
    df.to_csv("qualitative_results.csv", index=False)
    print("✅ Saved qualitative results to qualitative_results.csv")

if __name__ == "__main__":
    main()
