import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import csv
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

def benchmark_inference(model, dataloader, device, num_batches=20):
    times = []
    with torch.no_grad():
        for i, (image_emb, audio_emb, text_emb) in enumerate(dataloader):
            if i >= num_batches:
                break
            image_emb = image_emb.to(device)
            audio_emb = audio_emb.to(device)
            text_emb = text_emb.to(device)

            start_time = time.time()
            _ = model(image_emb, audio_emb, text_emb)
            end_time = time.time()

            elapsed = (end_time - start_time) * 1000  # milliseconds
            times.append(elapsed)

    avg_time = sum(times) / len(times)
    return avg_time

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MultimodalTripletDataset(r'C:\Aneesh\EDI VI\data\processed\multimodal_triplets.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    img_emb, audio_emb, text_emb = dataset[0]
    input_dims = (img_emb.shape[-1], audio_emb.shape[-1], text_emb.shape[-1])

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

    with open("inference_speed_results.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Avg Inference Time (ms) per Batch"])

        for name, path in model_paths.items():
            print(f"Benchmarking {name}...")
            model = load_model(model_classes[name], input_dims, path, device)
            avg_time = benchmark_inference(model, dataloader, device)
            writer.writerow([name, f"{avg_time:.2f}"])
            print(f"{name}: {avg_time:.2f} ms/batch")

    print("✅ Benchmarking complete. Results saved to inference_speed_results.csv")

if __name__ == "__main__":
    main()
