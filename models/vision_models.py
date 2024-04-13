import torch.nn as nn


class LSTM(nn.Module):
    """Model applies one LSTM layer followed by a FC layer"""
    def __init__(self, input_emb_size = 768, no_of_frames = 100):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_emb_size, 128)
        self.fc = nn.Linear(128*no_of_frames, 2)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 