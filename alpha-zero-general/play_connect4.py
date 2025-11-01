import argparse
import numpy as np

from Arena import Arena
from MCTS import MCTS
from utils import dotdict
from connect4.Connect4Game import Connect4Game
from connect4.Connect4Players import HumanConnect4Player
from connect4.pytorch.NNet import NNetWrapper as NNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--win-length", type=int, default=4)
    parser.add_argument("--checkpoint-folder", type=str, default="./temp/")
    parser.add_argument("--checkpoint-file", type=str, default="best.pth.tar")
    parser.add_argument("--num-mcts-sims", type=int, default=50)
    parser.add_argument("--cpuct", type=float, default=1.0)
    parser.add_argument("--human-vs-cpu", action="store_true")
    parser.add_argument("--cpu-vs-cpu", action="store_true")
    parser.add_argument("--human-first", action="store_true")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args_cli = parser.parse_args()

    g = Connect4Game(height=args_cli.height, width=args_cli.width, win_length=args_cli.win_length)

    n1 = NNet(g)
    n1.load_checkpoint(args_cli.checkpoint_folder, args_cli.checkpoint_file)

    mcts_args = dotdict({
        'numMCTSSims': int(args_cli.num_mcts_sims),
        'cpuct': float(args_cli.cpuct)
    })

    mcts1 = MCTS(g, n1, mcts_args)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    def human_wrap(play_fn):
        def _play(board):
            print("\n[Human input - canonical board (you are 1)]")
            Connect4Game.display(board)
            return play_fn(board)
        return _play

    def ai_wrap(policy_fn):
        def _play(board):
            print("\n[AI input - canonical board (AI is 1)]")
            Connect4Game.display(board)
            return policy_fn(board)
        return _play

    if args_cli.human_vs_cpu and args_cli.cpu_vs_cpu:
        raise SystemExit("Specify only one of --human-vs-cpu or --cpu-vs-cpu")

    human_is_p1 = None
    if args_cli.human_vs_cpu or not args_cli.cpu_vs_cpu:
        hp = human_wrap(HumanConnect4Player(g).play)
        if args_cli.human_first:
            p1, p2 = hp, ai_wrap(n1p)
            human_is_p1 = True
        else:
            p1, p2 = ai_wrap(n1p), hp
            human_is_p1 = False
    else:
        n2 = NNet(g)
        n2.load_checkpoint(args_cli.checkpoint_folder, args_cli.checkpoint_file)
        mcts2 = MCTS(g, n2, mcts_args)
        n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
        p1, p2 = ai_wrap(n1p), ai_wrap(n2p)

    arena = Arena(p1, p2, g, display=Connect4Game.display)

    # Show 2D board output by default in human-vs-cpu mode
    effective_verbose = args_cli.verbose or args_cli.human_vs_cpu

    if args_cli.games == 1:
        result = arena.playGame(verbose=effective_verbose)
        if args_cli.human_vs_cpu or human_is_p1 is not None:
            # result: 1 => Player1 wins, -1 => Player2 wins, ~0 => draw
            if result == 1:
                winner = "Human" if human_is_p1 else "AI"
            elif result == -1:
                winner = "AI" if human_is_p1 else "Human"
            else:
                winner = "Draw"
            print(f"Result: {result}  |  Winner: {winner}")
        else:
            print("Result:", result)
    else:
        one, two, draws = arena.playGames(args_cli.games, verbose=effective_verbose)
        print("P1 wins:", one, "P2 wins:", two, "Draws:", draws)


if __name__ == "__main__":
    main()
