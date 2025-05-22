import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import umap.umap_ as umap
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

def get_embeddings(model, dataloader, device, audio_proj, max_batches=5):
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (image_emb, audio_emb, text_emb) in enumerate(dataloader):
            if i >= max_batches:
                break

            image = image_emb.squeeze(1).to(device)
            audio = audio_emb.squeeze(1).to(device)
            text = text_emb.squeeze(1).to(device)

            audio_mapped = audio_proj(audio)
            target = (image + audio_mapped + text) / 3.0

            output = model(image_emb.to(device), audio_emb.to(device), text_emb.to(device))
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    return all_preds, all_targets

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

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    plt.figure(figsize=(12, 10))

    for i, (name, path) in enumerate(model_paths.items()):
        model = load_model(model_classes[name], input_dims, path, device)
        preds, targets = get_embeddings(model, dataloader, device, audio_proj)
        all_vectors = preds + targets
        all_labels = [f"{name} - Pred"] * len(preds) + [f"{name} - Target"] * len(targets)

        embedding_2d = reducer.fit_transform(all_vectors)

        x = embedding_2d[:len(preds)]
        y = embedding_2d[len(preds):]

        plt.scatter(x[:, 0], x[:, 1], label=f'{name} - Predicted', alpha=0.6, marker='o')
        plt.scatter(y[:, 0], y[:, 1], label=f'{name} - Target', alpha=0.6, marker='x')

    plt.legend()
    plt.title("UMAP Projection of Predicted vs Target Embeddings (All Models)")
    plt.savefig("embedding_visualization.png")
    print("✅ Saved UMAP plot to embedding_visualization.png")

if __name__ == "__main__":
    main()
