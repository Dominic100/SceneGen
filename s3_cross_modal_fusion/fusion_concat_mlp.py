import torch
import torch.nn as nn

class ConcatMLPFusion(nn.Module):
    def __init__(self, input_dims, output_dim=512):
        super().__init__()
        image_dim, audio_dim, text_dim = input_dims
        input_dim = image_dim + audio_dim + text_dim  # 512 + 2048 + 512 = 3072

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, image_emb, audio_emb, text_emb):
        # Input: (B, 1, D) → squeeze to (B, D)
        image_emb = image_emb.squeeze(1)
        audio_emb = audio_emb.squeeze(1)
        text_emb = text_emb.squeeze(1)

        # Concatenate: (B, 3072)
        x = torch.cat((image_emb, audio_emb, text_emb), dim=1)
        return self.mlp(x)
