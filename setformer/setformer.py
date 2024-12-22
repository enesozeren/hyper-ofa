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
            nn.Linear(emb_dim, 2*emb_dim),
            nn.GELU(),
            nn.Linear(2*emb_dim, output_dim),
            nn.GELU()
        )

        # Output scaling layer
        self.output_scale = nn.Parameter(torch.tensor(0.001))  # Initialize with a small scale

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
        
        # Get the CLS token in the first dimension
        x = x[:, 0, :]
        
        # Feed the CLS token to the output layer
        x = self.output_layers(x) # (batch_size, output_dim)
        
        # Apply scaling to the output since target outputs are very small (around e-5)
        x = self.output_scale * x

        return x