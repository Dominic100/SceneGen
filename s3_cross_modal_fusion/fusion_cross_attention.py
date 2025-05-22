import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, image_dim=512, audio_dim=2048, text_dim=512, output_dim=512, num_heads=4):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.audio_proj = nn.Linear(audio_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, image_emb, audio_emb, text_emb):
        # Squeeze from (B, 1, D) → (B, D)
        image = self.image_proj(image_emb.squeeze(1))  # (B, output_dim)
        audio = self.audio_proj(audio_emb.squeeze(1))  # (B, output_dim)
        text = self.text_proj(text_emb.squeeze(1))     # (B, output_dim)

        # Stack image and audio as KV
        kv = torch.stack([image, audio], dim=1)  # (B, 2, output_dim)
        query = text.unsqueeze(1)                # (B, 1, output_dim)

        # Cross-attention: text attends to image+audio
        attended, _ = self.cross_attn(query, kv, kv)  # (B, 1, output_dim)
        return self.fc(attended.squeeze(1))           # (B, output_dim)
