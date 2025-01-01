import torch
import torch.nn as nn
import torch.nn.functional as F

class SetFormer(nn.Module):
    def __init__(self, emb_dim, num_heads, num_layers, dim_feedforward, 
                 output_dim, context_size, dropout, word_vector_emb, padding_idx):
        '''
        :param emb_dim: The dimension of the input word vector
        :param num_heads: The number of heads in the multihead attention
        :param num_layers: The number of layers in the transformer encoder
        :param dim_feedforward: The hidden dimension of the transformer encoder
        :param output_dim: The dimension of the output
        :param context_size: The size of the context window
        :param dropout: The dropout rate
        :param word_vector_emb: The word vector embeddings (includs the CLS token at the start)
        :param padding_idx: The index of the padding token in the word vector embeddings
        '''
        super(SetFormer, self).__init__()

        self.context_size = context_size
        self.padding_idx = padding_idx
        
        # The external word vectors will be the embedding layer and will be frozen
        self.word_vector_emb_layer = nn.Embedding.from_pretrained(embeddings=word_vector_emb, 
                                                                  freeze=True, 
                                                                  padding_idx=padding_idx)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        
        # An Encoder block to process the input
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.encoder_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers for the CLS token
        self.output_layers = nn.Sequential(
            nn.Linear(emb_dim, 2*output_dim),
            nn.GELU(),
            nn.Linear(2*output_dim, output_dim),
            nn.GELU()
        )

    def forward(self, x):
        '''
        :param x: The input tensor of shape (batch_size, context_size)
        '''
        # Create a padding mask
        padding_mask = (x == self.padding_idx)
    
        # Get the embeddings
        x = self.word_vector_emb_layer(x) # (batch_size, context_size, emb_dim)
        
        # Apply dropout
        x = self.dropout(x) # (batch_size, context_size, emb_dim)
        
        # Pass the embeddings through the transformer encoder
        x = self.encoder_block(x, src_key_padding_mask=padding_mask) # (batch_size, context_size, emb_dim)
        
        # Compute mean pooling across the sequence, excluding padding tokens
        # Mask the padding tokens by setting them to zero
        x_masked = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # (batch_size, context_size, emb_dim)

        # Count the valid tokens (non-padding tokens) for each sequence
        valid_token_counts = (~padding_mask).sum(dim=1).unsqueeze(-1).clamp(min=1)  # (batch_size, 1)

        # Compute the mean of non-padding tokens
        x = x_masked.sum(dim=1) / valid_token_counts  # (batch_size, emb_dim)

        # Feed the mean pooled representation to the output layer
        x = self.output_layers(x)  # (batch_size, output_dim)

        return x