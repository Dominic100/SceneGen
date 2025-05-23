import torch
import torch.nn as nn

class SceneTransformerEncoder(nn.Module):
    def __init__(
        self,
        img_emb_dim=512,
        audio_emb_dim=2048,
        text_emb_dim=512,
        fusion_emb_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_seq_len=5,
        output_dim=512,
    ):
        super(SceneTransformerEncoder, self).__init__()

        self.max_seq_len = max_seq_len

        # Projection layers for each modality to fusion_emb_dim
        self.img_proj = nn.Linear(img_emb_dim, fusion_emb_dim)
        self.audio_proj = nn.Linear(audio_emb_dim, fusion_emb_dim)
        self.text_proj = nn.Linear(text_emb_dim, fusion_emb_dim)

        # Positional encoding (learned)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, fusion_emb_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_emb_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection to output embedding
        self.output_proj = nn.Linear(fusion_emb_dim, output_dim)

    def forward(self, img_seq, audio_seq, text_seq):
        """
        Handle the specific input shape [B, T, 1, D] coming from the dataset.
        """
        # Print shapes for debugging
        print(f"Input shapes: img={img_seq.shape}, audio={audio_seq.shape}, text={text_seq.shape}")
        
        # Handle the actual shape: [B, T, 1, D]
        if len(img_seq.shape) == 4 and img_seq.shape[2] == 1:
            # Squeeze out the extra dimension to get [B, T, D]
            img_seq = img_seq.squeeze(2)
            audio_seq = audio_seq.squeeze(2)
            text_seq = text_seq.squeeze(2)
            
        # Now we should have tensors with shape [B, T, D]
        B, T, D_img = img_seq.shape
        _, _, D_audio = audio_seq.shape
        _, _, D_text = text_seq.shape
        
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        
        # Project each modality
        img_proj = self.img_proj(img_seq)      # (B, T, fusion_emb_dim)
        audio_proj = self.audio_proj(audio_seq)  # (B, T, fusion_emb_dim)
        text_proj = self.text_proj(text_seq)    # (B, T, fusion_emb_dim)

        # Simple fusion by summation
        fused = img_proj + audio_proj + text_proj  # (B, T, fusion_emb_dim)

        # Add positional encoding (broadcast over batch)
        pos_emb = self.pos_embedding[:T, :].unsqueeze(0).expand(B, -1, -1)  # (B, T, fusion_emb_dim)
        fused = fused + pos_emb

        # Transformer encoding
        x = self.transformer_encoder(fused)  # (B, T, fusion_emb_dim)

        # Pool over temporal dimension (mean pooling)
        pooled = x.mean(dim=1)  # (B, fusion_emb_dim)

        # Final output projection
        out = self.output_proj(pooled)  # (B, output_dim)

        # Normalize embedding vector (optional but useful for contrastive)
        out = nn.functional.normalize(out, p=2, dim=1)

        return out