# connect4/mcts.py
import math, numpy as np
import torch

class MCTS:
    def __init__(self, game_cls, net, sims=200, cpuct=1.25, device="cpu"):
        self.game_cls, self.net = game_cls, net
        self.sims, self.cpuct, self.device = sims, cpuct, device
        self.Q, self.N, self.P, self.children = {}, {}, {}, {}

    def policy(self, game):
        # return improved policy via visit counts
        s_key = self.key(game)
        for _ in range(self.sims):
            self.search(game.clone())
        Nsa = self.N.get(s_key, {})
        pi = np.zeros(7, dtype=np.float32)
        for a, n in Nsa.items(): pi[a] = n
        if pi.sum() == 0: pi[:] = 1.0
        return pi / pi.sum()

    # ---- internals ----
    def key(self, game): return game.board.tobytes() + bytes([int(game.player==1)])

    def search(self, game):
        s_key = self.key(game)
        done, z = game.is_terminal()
        if done:
            return -z  # value from current player's view

        if s_key not in self.P:
            # expand
            x = torch.from_numpy(game.canonical_obs()).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p_logits, v = self.net(x)
                p = torch.softmax(p_logits, dim=-1).squeeze(0).cpu().numpy()
            valid = game.valid_moves()
            p = p * valid; s = p.sum(); p = p/s if s>0 else valid/valid.sum()
            self.P[s_key] = p
            self.N[s_key], self.Q[s_key] = {}, {}
            return -float(v.item())

        # select
        best_a, best_u = -1, -1e9
        total_N = sum(self.N[s_key].values()) if self.N[s_key] else 1
        for a in range(7):
            if game.valid_moves()[a] == 0: continue
            q = self.Q[s_key].get(a, 0.0)
            n = self.N[s_key].get(a, 0)
            u = q + self.cpuct * self.P[s_key][a] * math.sqrt(total_N + 1) / (1 + n)
            if u > best_u: best_u, best_a = u, a

        game.play(best_a)
        v = self.search(game)

        # backup
        q, n = self.Q[s_key].get(best_a, 0.0), self.N[s_key].get(best_a, 0)
        new_q = (n*q + v) / (n+1)
        self.Q[s_key][best_a], self.N[s_key][best_a] = new_q, n+1
        return -v
