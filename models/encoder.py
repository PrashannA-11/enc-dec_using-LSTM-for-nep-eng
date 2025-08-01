# models/encoder.py

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src)  # [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell
