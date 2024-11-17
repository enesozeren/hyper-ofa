import torch
import torch.nn as nn
import torch.nn.functional as F

class SetFormer(nn.Module):
    def __init__(self, emb_dim, num_heads, num_layers, dim_feedforward, output_dim, context_size, dropout, word_vector_emb):
        '''
        :param emb_dim: The dimension of the input word vector
        :param num_heads: The number of heads in the multihead attention
        :param num_layers: The number of layers in the transformer encoder
        :param hidden_dim: The hidden dimension of the transformer encoder
        :param output_dim: The dimension of the output
        :param context_size: The size of the context window
        :param dropout: The dropout rate
        :param word_vector_emb: The word vector embeddings (includs the CLS token at the end)
        '''
        super(SetFormer, self).__init__()

        self.context_size = context_size
        # The external word vectors will be the embedding layer and will be frozen
        self.word_vector_emb_layer = nn.Embedding.from_pretrained(embeddings=word_vector_emb, freeze=True)
        # An Encoder block to process the input
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.encoder_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output layer for the CLS token
        self.output_layer = nn.Linear(emb_dim, output_dim)

    def forward(self, x):
        # Use batch first

        # x is a tensor of shape (batch_size, context_size)
        x = self.word_vector_emb_layer(x) # (batch_size, context_size, emb_dim)
        x = self.encoder_block(x) # (batch_size, context_size, emb_dim)
        # Get the CLS token in the first dimension
        x = x[:, 0, :]
        # Feed the CLS token to the output layer
        x = self.output_layer(x) # (batch_size, output_dim)

        return x