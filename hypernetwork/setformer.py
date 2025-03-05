import torch
import torch.nn as nn

class SetFormer(nn.Module):
    def __init__(self, emb_dim, num_heads, num_layers, output_dim, 
                    context_size, dropout, word_vector_emb, padding_idx):
        super(SetFormer, self).__init__()

        self.context_size = context_size
        self.padding_idx = padding_idx
        
        # Embedding Layer
        self.word_vector_emb_layer = nn.Embedding.from_pretrained(
            embeddings=word_vector_emb, 
            freeze=True, 
            padding_idx=padding_idx
        )
        
        # Second Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=num_heads, 
            dim_feedforward=8*emb_dim,
            dropout=dropout, 
            batch_first=True
        )
        self.encoder_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output linear layers
        self.linear_output_layers = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(2 * emb_dim, output_dim)
        )

    def forward(self, x):
        """
        :param x: The input tensor of shape (batch_size, context_size)
        """
        # Create a padding mask
        padding_mask = (x == self.padding_idx)
    
        # Get the embeddings
        x = self.word_vector_emb_layer(x)  # (batch_size, context_size, emb_dim)
        
        # Second Transformer Encoder Block
        x = self.encoder_block(x, src_key_padding_mask=padding_mask)  # (batch_size, context_size, emb_dim)

        # Compute mean pooling across the sequence, excluding padding tokens
        # Mask the padding tokens by setting them to zero
        x_masked = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # (batch_size, context_size, emb_dim)
        # Count the valid tokens (non-padding tokens) for each sequence
        valid_token_counts = (~padding_mask).sum(dim=1).unsqueeze(-1).clamp(min=1)  # (batch_size, 1)
        # Compute the mean of non-padding tokens
        x = x_masked.sum(dim=1) / valid_token_counts  # (batch_size, emb_dim)

        # Output
        x = self.linear_output_layers(x)  # (batch_size, output_dim)

        return x