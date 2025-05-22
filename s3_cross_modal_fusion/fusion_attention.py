# fusion_attention.py

import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, image_dim, audio_dim, text_dim, output_dim=512):
        super().__init__()
        
        # Linear projections for each modality
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.audio_proj = nn.Linear(audio_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)

        # Attention mechanism
        self.attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=4, batch_first=True)

        # Final output layer (optional, here just identity)
        self.output_proj = nn.Identity()

    def forward(self, image_emb, audio_emb, text_emb):
        # Input shape: (B, 1, D) → squeeze to (B, D)
        image = self.image_proj(image_emb.squeeze(1))  # (B, output_dim)
        audio = self.audio_proj(audio_emb.squeeze(1))  # (B, output_dim)
        text = self.text_proj(text_emb.squeeze(1))     # (B, output_dim)

        # Stack into sequence for attention
        x = torch.stack([image, audio, text], dim=1)   # (B, 3, output_dim)

        # Self-attention
        attn_output, _ = self.attn(x, x, x)            # (B, 3, output_dim)

        # Pool (mean across sequence length)
        fused = attn_output.mean(dim=1)                # (B, output_dim)

        return self.output_proj(fused)

    def project_modalities(self, image_emb, audio_emb, text_emb):
        # Used to compute average projection as a training target
        image = self.image_proj(image_emb.squeeze(1))
        audio = self.audio_proj(audio_emb.squeeze(1))
        text = self.text_proj(text_emb.squeeze(1))
        return (image + audio + text) / 3.0
