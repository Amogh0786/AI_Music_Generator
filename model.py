import torch
import torch.nn as nn

class MusicLSTM(nn.Module):
    def __init__(self, n_vocab, embed_dim=256, hidden_dim=512, num_layers=3):
        super(MusicLSTM, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, n_vocab)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, sequence_length, embed_dim)
        lstm_out, _ = self.lstm(embedded)
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        
        # We only want the output from the last time step
        last_time_step = lstm_out[:, -1, :]
        
        out = self.fc(last_time_step)
        return out
