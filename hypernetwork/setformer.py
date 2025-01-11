import torch
import torch.nn as nn

class VectorSpaceTransformLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VectorSpaceTransformLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2 * output_dim)
        self.fc2 = nn.Linear(2 * output_dim, output_dim)
        self.skip_connection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else lambda x: x
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        """
        :param x: Input tensor of shape (batch_size, context_size, input_dim)
        :param mask: Optional padding mask of shape (batch_size, context_size)
        :return: Transformed tensor of shape (batch_size, context_size, output_dim)
        """
        skip = self.skip_connection(x)  # Skip connection
        x = self.fc1(x)  # First transformation
        x = self.activation(x)  # Apply non-linearity
        x = self.fc2(x)  # Second transformation
        x = x + skip  # Add skip connection

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)  # Mask at the VERY end

        return x

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
        
        # Vector Space Transform Layer
        self.vector_space_transform_layer = VectorSpaceTransformLayer(
            input_dim=emb_dim,
            output_dim=output_dim
        )
        
        # Second Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim, 
            nhead=num_heads, 
            dim_feedforward=4 * output_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        :param x: The input tensor of shape (batch_size, context_size)
        """
        # Create a padding mask
        padding_mask = (x == self.padding_idx)
    
        # Get the embeddings
        x = self.word_vector_emb_layer(x)  # (batch_size, context_size, emb_dim)
        
        # Transform to output_dim
        x = self.vector_space_transform_layer(x, mask=padding_mask)  # (batch_size, context_size, output_dim)
        
        # Second Transformer Encoder Block
        x = self.encoder_block(x, src_key_padding_mask=padding_mask)  # (batch_size, context_size, output_dim)

        # Compute mean pooling across the sequence, excluding padding tokens
        # Mask the padding tokens by setting them to zero
        x_masked = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # (batch_size, context_size, emb_dim)
        # Count the valid tokens (non-padding tokens) for each sequence
        valid_token_counts = (~padding_mask).sum(dim=1).unsqueeze(-1).clamp(min=1)  # (batch_size, 1)
        # Compute the mean of non-padding tokens
        x = x_masked.sum(dim=1) / valid_token_counts  # (batch_size, emb_dim)

        return x