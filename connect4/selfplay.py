# connect4/selfplay.py  (outline)
import random, torch, torch.optim as optim
from collections import deque
import numpy as np

def play_one_game(Game, mcts, temperature_moves=10):
    data = []  # list of (obs, pi, player)
    g = Game()
    while True:
        obs = g.canonical_obs()
        pi = mcts.policy(g)                 # run K sims
        # sample in opening, argmax later
        if sum(sum(g.board==0)) >= 35 and temperature_moves>0:
            a = np.random.choice(7, p=pi); temperature_moves -= 1
        else:
            a = int(np.argmax(pi))
        data.append((obs, pi, g.player))
        g.play(a)
        done, z = g.is_terminal()
        if done:
            # assign outcomes from each state’s player perspective
            out = []
            for (obs_i, pi_i, player_i) in data:
                out.append((obs_i, pi_i, z if player_i==g.player*-1 else -z))
            return out

def train_epoch(net, batch, lr=1e-3, wd=1e-4):
    net.train()
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    obs, pis, zs = batch  # tensors
    p_logits, v = net(obs)
    policy_loss = torch.nn.functional.cross_entropy(p_logits, torch.argmax(pis, dim=1))
    value_loss = torch.nn.functional.mse_loss(v, zs)
    loss = policy_loss + value_loss
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())
