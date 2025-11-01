# connect4/nnet.py
import torch, torch.nn as nn, torch.nn.functional as F

class AZNet(nn.Module):
    def __init__(self, chans=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, chans, 3, padding=1), nn.ReLU(),
            nn.Conv2d(chans, chans, 3, padding=1), nn.ReLU(),
            nn.Conv2d(chans, chans, 3, padding=1), nn.ReLU(),
        )
        # policy head
        self.p_conv = nn.Conv2d(chans, 2, 1); self.p_fc = nn.Linear(2*6*7, 7)
        # value head
        self.v_conv = nn.Conv2d(chans, 1, 1); self.v_fc1 = nn.Linear(1*6*7, 64); self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)                      # [B,C,6,7]
        p = F.relu(self.p_conv(h)).view(x.size(0), -1)
        p = self.p_fc(p)                      # [B,7] (logits)
        v = F.relu(self.v_conv(h)).view(x.size(0), -1)
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v))))  # [B,1]
        return p, v.squeeze(-1)
