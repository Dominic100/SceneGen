import torch
import torch.nn as nn

class TensorFusion(nn.Module):
    def __init__(self, image_dim=512, audio_dim=2048, text_dim=512, output_dim=512):
        super().__init__()
        # Linear projections for each modality
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.audio_proj = nn.Linear(audio_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)

        # Bilinear interaction layer (outer product style fusion)
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, image_emb, audio_emb, text_emb):
        image = self.image_proj(image_emb.squeeze(1))  # (B, D)
        audio = self.audio_proj(audio_emb.squeeze(1))  # (B, D)
        text = self.text_proj(text_emb.squeeze(1))     # (B, D)

        # Element-wise multiply image & audio → (B, D)
        ia_fusion = image * audio

        # Outer product: fuse (ia_fusion) with text → (B, D, D)
        outer = torch.bmm(ia_fusion.unsqueeze(2), text.unsqueeze(1))  # (B, D, D)
        fused = outer.view(outer.size(0), -1)  # Flatten to (B, D*D)

        return self.fusion_layer(fused)  # Output: (B, output_dim)
