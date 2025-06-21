import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCModel(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
