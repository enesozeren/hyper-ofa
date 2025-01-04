import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers, output_dim, 
                 context_size, dropout, word_vector_emb, padding_idx):
        '''
        :param emb_dim: The dimension of the input word vector
        :param hidden_dim: The hidden dimension of the LSTM
        :param num_layers: The number of layers in the LSTM
        :param output_dim: The dimension of the output
        :param context_size: The size of the context window
        :param dropout: The dropout rate
        :param padding_idx: The index of the padding token in the word vector embeddings
        '''
        super(LSTMModel, self).__init__()

        self.context_size = context_size
        self.padding_idx = padding_idx
        
        # The external word vectors will be the embedding layer and will be frozen
        self.word_vector_emb_layer = nn.Embedding.from_pretrained(embeddings=word_vector_emb, 
                                                                  freeze=True, 
                                                                  padding_idx=padding_idx)
        
        # Bidirectional LSTM block
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, dropout=dropout, 
                            bidirectional=True)
        
        # Output linear layers
        self.linear_output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * output_dim),
            nn.GELU(),
            nn.Linear(2 * output_dim, output_dim)
        )

    def forward(self, x):
        '''
        :param x: The input tensor of shape (batch_size, context_size)
        '''
        # Create a padding mask
        padding_mask = (x == self.padding_idx)
    
        # Get the embeddings
        x = self.word_vector_emb_layer(x)  # (batch_size, context_size, emb_dim)
        
        # Compute the lengths tensor and move it to the CPU
        lengths = torch.sum(~padding_mask, dim=1).cpu().to(torch.int64)  # Ensure it's a 1D CPU int64 tensor
        # Pass the embeddings through the LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, 
                                                         batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # We use the last hidden state for output
        # h_n contains the hidden state for both directions
        # Concatenate the forward and backward hidden states
        h_n_bidirectional = torch.cat((h_n[-2], h_n[-1]), dim=-1)  # Concatenate last forward and backward states
        x = self.linear_output_layers(h_n_bidirectional)  # (batch_size, output_dim)

        return x