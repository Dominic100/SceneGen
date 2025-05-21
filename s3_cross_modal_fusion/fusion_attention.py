import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, image_dim, audio_dim, text_dim, hidden_dim=512, output_dim=512):
        super().__init__()

        # Project all modalities to same hidden_dim
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Attention weights over 3 modalities
        self.attn_weights = nn.Parameter(torch.ones(3))  # Learnable weights

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, image_emb, audio_emb, text_emb):
        image_emb = image_emb.squeeze(1)  # (B, 512)
        audio_emb = audio_emb.squeeze(1)  # (B, 2048)
        text_emb = text_emb.squeeze(1)    # (B, 512)

        # Project to common space
        image_proj = self.image_proj(image_emb)  # (B, hidden)
        audio_proj = self.audio_proj(audio_emb)
        text_proj = self.text_proj(text_emb)

        # Stack for attention
        stacked = torch.stack([image_proj, audio_proj, text_proj], dim=1)  # (B, 3, hidden)

        # Softmax over learnable weights
        attn = F.softmax(self.attn_weights, dim=0)  # (3,)
        attn = attn.unsqueeze(0).unsqueeze(-1)      # (1, 3, 1)

        # Apply attention weights
        fused = torch.sum(attn * stacked, dim=1)    # (B, hidden)

        # Final projection
        return self.output_proj(fused)              # (B, output_dim)
