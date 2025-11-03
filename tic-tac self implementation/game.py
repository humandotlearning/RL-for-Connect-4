import numpy as np
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TTState:
    board: np.ndarray
    to_play: int   # values can -1,1 

    @staticmethod
    def new():
        return TTState(np.zeros((3,3), dtype=int), 1)

    def legal_moves(self):
        return [ r*3 + c for r in range(3) for c in range(3) if self.board[r,c] == 0 ]

    def apply(self, move):
        r,c = divmod(move,3)
        assert self.board[r,c] == 0, f"Move {move} is not legal"
        b = self.board.copy()
        b[r,c] = self.to_play
        return TTState(b, -self.to_play)  # - sign is to toggle the next player to play

    def get_valid_moves(self):
        return self.legal_moves()

    def winner(self):
        b = self.board
        lines = [ sum(b[0]), sum(b[1]), sum(b[2]), sum(b[:,0]), sum(b[:,1]), sum(b[:,2]), sum(np.diag(b)), sum(np.diag(np.fliplr(b))) ]
        if 3 in lines:
            return 1
        if -3 in lines:
            return -1
        if 0 in self.board:
            return None
        return 0  # board is full

    def terminal(self):
        return self.winner() is not None

if __name__ == "__main__":
    s = TTState.new()
    while not s.terminal():
        print(s.board)
        print("Valid moves: ", s.get_valid_moves())
        move = int(input("Move: "))
        s = s.apply(move)
    print(s.board)
    print("Winner: ", s.winner())

    
    