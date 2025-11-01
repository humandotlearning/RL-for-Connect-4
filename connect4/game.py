# connect4/game.py
import numpy as np

ROWS, COLS = 6, 7

class Connect4:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)  # 0 empty, +1/-1 stones
        self.player = 1  # +1 to move

    def clone(self):
        g = Connect4()
        g.board = self.board.copy()
        g.player = self.player
        return g

    def valid_moves(self):
        v = np.zeros(COLS, dtype=np.int8)
        for c in range(COLS):
            v[c] = int(self.board[0, c] == 0)
        return v

    def play(self, col):
        # drop piece to lowest empty row in col
        for r in range(ROWS-1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = self.player
                self.player *= -1
                return
        raise ValueError("Column full")

    def is_terminal(self):
        # check win/draw; return (done, result_from_current_player_perspective_after_switch)
        # Implement 4-in-a-row scan; if no empty -> draw (0).
        # If last mover (opponent of self.player) has 4, result = -1 (bad for current).
        ...
    
    def canonical_obs(self):
        cur = (self.board == self.player).astype(np.float32)
        opp = (self.board == -self.player).astype(np.float32)
        turn = np.full_like(cur, 1.0 if self.player == 1 else 0.0)
        return np.stack([cur, opp, turn], axis=0)  # [3,6,7]
