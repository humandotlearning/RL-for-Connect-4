import os

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Dataset, DataLoader

from NeuralNet import NeuralNet


# pre activation renet block
# pre activation is better because it is more stable
# pre-act keeps the skip truly identity and gives cleaner gradient flow; empirically it eases optimization and improves generalization vs. post-act.
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        x = self.dropout(x)
        x += identity
        return x


class Connect4Net(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        channels = args.num_channels
        n_resblocks = getattr(args, 'num_residual_layers', 5)

        # single input channel (canonical board)
        self.conv0 = nn.Conv2d(1, channels, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(channels)

        self.resblocks = nn.Sequential(
            *[ResidualBlock(channels, dropout=args.dropout) for _ in range(n_resblocks)]
        )

        # policy head
        self.p_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

        # value head
        self.v_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(1 * self.board_x * self.board_y, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, s):
        # s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x board_x x board_y
        x = F.relu(self.bn0(self.conv0(s)))
        x = self.resblocks(x)

        # policy head
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)
        p = F.log_softmax(p, dim=1)

        # value head
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = self.v_fc1(v)
        v = F.relu(v)
        v = self.v_fc2(v)
        v = torch.tanh(v)

        return p, v




#############################################################
# this code part will not be used
class AZDataset(Dataset):
    def __init__(self, data):
        self.boards,  self.pis, self.vs = [], [], []
        for board, pi, v in data:
            board = np.array(board)
            cur = (board ==1).astype(np.float32)
            opp = (board ==-1).astype(np.float32)
            planes = np.stack([cur, opp], axis=0)
            self.boards.append(planes)
            self.pis.append(np.array(pi, dtype=np.float32))
            self.vs.append(np.array(v, dtype=np.float32))
        self.boards = np.array(self.boards)
        self.pis = np.array(self.pis)
        self.vs = np.array(self.vs)

    
    def __len__(self):
        return len(self.boards)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.boards[idx], dtype=torch.float32), 
            torch.tensor(self.pis[idx], dtype=torch.float32), 
            torch.tensor(self.vs[idx], dtype=torch.float32)
        )


##############################################################