import os
import sys
import time
import argparse
import numpy as np
from playwright.sync_api import sync_playwright

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ALPHA_DIR = os.path.join(BASE_DIR, "alpha-zero-general")
if ALPHA_DIR not in sys.path:
    sys.path.insert(0, ALPHA_DIR)

from connect4.Connect4Game import Connect4Game
from connect4.pytorch.NNet import NNetWrapper as NNet
from MCTS import MCTS
from utils import dotdict

from .papergames_adapter import PapergamesConnect4

def choose_ai_move(g, mcts, board_np, cur_player):
    """
    Choose AI move using MCTS.

    IMPROVEMENT: Tree reuse is naturally happening here
    - The same MCTS instance is used throughout the game
    - Search statistics (Q-values, visit counts) accumulate across moves
    - This makes the agent stronger as the game progresses
    - Note: temp=0 means greedy selection (best move), no exploration

    Future optimization: Could implement explicit tree pruning to remove
    branches that are no longer reachable after opponent moves
    """
    canonical = g.getCanonicalForm(board_np, cur_player)
    # No Dirichlet noise for online play - we want deterministic strong play
    pi = mcts.getActionProb(canonical, temp=0, add_dirichlet_noise=False)
    return int(np.argmax(pi))

def report_game_end(g, board_np, cur_player, last_mover: str) -> bool:
    """Return True if game ended and log winner based on who moved last.
    getGameEnded(board, player_to_move_next) returns:
      +1 if player_to_move_next is winner
      -1 if the other (last mover) is winner
      1e-4 for draw
    Therefore:
      - After AI move (last_mover='AI', player_to_move_next=opponent), AI wins -> res < 0
      - After Opponent move (last_mover='Opponent', player_to_move_next=AI), AI wins -> res > 0
    """
    res = g.getGameEnded(board_np, cur_player)
    if res == 0:
        return False
    if abs(res) < 1e-4:
        log("Game over: Draw")
        return True
    if last_mover == "AI":
        ai_wins = res < 0
    else:  # Opponent
        ai_wins = res > 0
    log("Game over: AI wins" if ai_wins else "Game over: Opponent wins")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "robot", "friend-create", "friend-join"], default="random")
    ap.add_argument("--nickname", type=str, default="agent")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--slowmo", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--checkpoint-folder", type=str, default=os.path.join(BASE_DIR, "temp"))
    ap.add_argument("--checkpoint-file", type=str, default="best.pth.tar")
    ap.add_argument("--num-mcts-sims", type=int, default=100)
    ap.add_argument("--cpuct", type=float, default=1.0)
    ap.add_argument("--room-url", type=str, default=None, help="Friend room URL for friend-join mode")
    ap.add_argument("--friend-wait-timeout", type=int, default=180, help="Seconds to wait for friend to join/start before giving up")
    args = ap.parse_args()

    g = Connect4Game()
    nnet = NNet(g)
    try:
        nnet.load_checkpoint(args.checkpoint_folder, args.checkpoint_file)
    except FileNotFoundError:
        log(f"Checkpoint not found: {os.path.join(args.checkpoint_folder, args.checkpoint_file)}")
        log("Provide a trained checkpoint via --checkpoint-folder/--checkpoint-file, or train a model to create one.")
        return 2
    mcts = MCTS(g, nnet, dotdict({"numMCTSSims": int(args.num_mcts_sims), "cpuct": float(args.cpuct)}))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=bool(args.headless), slow_mo=int(args.slowmo))
        ctx = browser.new_context()
        page = ctx.new_page()

        adapter = PapergamesConnect4(page)
        log(f"Navigating to papergames.io (mode={args.mode}, nickname={args.nickname})")
        if args.mode in ("random", "robot"):
            adapter.navigate(mode=args.mode, nickname=args.nickname)
        elif args.mode == "friend-create":
            url = adapter.create_friend_room(nickname=args.nickname)
            log(f"Friend room created. Share this URL with your friend: {url}")
        elif args.mode == "friend-join":
            if not args.room_url:
                log("--room-url is required for friend-join mode")
                return 3
            adapter.join_friend_room(args.room_url, nickname=args.nickname)
        if not adapter.wait_for_board(timeout_ms=args.timeout * 1000):
            log("Board not detected in time")
            return 1

        grid, _ = adapter.parse_grid()
        # IMPROVEMENT: Use same hashing method as wait_board_change for consistency
        last_hash = hash(str([tuple(row) for row in grid]))
        cur_player = 1  # track 'player to move next' in AlphaZero convention (+1 / -1)
        log("Board detected. Starting loop.")

        move_idx = 0

        # IMPROVEMENT: Simplified and more robust state machine for online play
        # The previous implementation had complex nested conditionals that could desync
        # New approach: Clear separation between color discovery, AI moves, and opponent moves

        while True:
            # Color Discovery Phase
            # FIX: Better detection of who goes first by checking initial board state
            if adapter.my_fill is None:
                board_np = np.zeros((6, 7), dtype=int)

                # Check if opponent has already moved (board is not empty)
                if np.any(grid):
                    log("Opponent moved first, detecting colors...")
                    adapter.update_fills_after_move([[None]*7 for _ in range(6)], grid)
                    log(f"Colors resolved: my_fill={adapter.my_fill} | opp_fill={adapter.opp_fill}")
                    board_np = np.array(adapter.grid_to_board(grid), dtype=int)
                    # We're second player, so continue to AI move
                else:
                    # We go first, make discovery move
                    log("Making first move for color discovery...")
                    col = 3  # Center column is strategically better than edge
                    log(f"Color discovery: clicking column {col}")
                    adapter.click_column(col)
                    new_grid, last_hash = adapter.wait_board_change(last_hash, timeout_s=args.timeout)
                    if not new_grid:
                        log("No board change after first move; stopping.")
                        break
                    adapter.update_fills_after_move(grid, new_grid)
                    grid = new_grid
                    cur_player *= -1  # Toggle to opponent
                    log(f"Colors resolved: my_fill={adapter.my_fill} | opp_fill={adapter.opp_fill}")
                    board_np = np.array(adapter.grid_to_board(grid), dtype=int)

                    # Check if game ended (shouldn't happen on first move, but be safe)
                    if report_game_end(g, board_np, cur_player, last_mover="AI"):
                        break

                    # Wait for opponent's response
                    new_grid, last_hash = adapter.wait_board_change(last_hash, timeout_s=args.timeout)
                    if new_grid:
                        grid = new_grid
                        cur_player *= -1  # Back to AI's turn
                        board_np = np.array(adapter.grid_to_board(grid), dtype=int)
                        if report_game_end(g, board_np, cur_player, last_mover="Opponent"):
                            break
                        move_idx += 1
                    else:
                        log("Opponent did not respond; stopping.")
                        break
                    continue

            # Main Game Loop - AI's Turn
            board_np = np.array(adapter.grid_to_board(grid), dtype=int)

            # IMPROVEMENT: Validate that it's actually our turn by checking valid moves
            canonical = g.getCanonicalForm(board_np, cur_player)
            valids = g.getValidMoves(canonical, 1)
            if not valids.any():
                log("No valid moves available; game should have ended.")
                break

            # Choose and make AI move
            col = choose_ai_move(g, mcts, board_np, cur_player)

            # IMPROVEMENT: Validate move is actually valid before clicking
            if not valids[col]:
                log(f"WARNING: MCTS chose invalid move {col}! Choosing first valid move.")
                col = int(np.where(valids)[0][0])

            log(f"AI move #{move_idx}: choosing column {col}")
            adapter.click_column(col)

            # Wait for AI move to register
            new_grid, last_hash = adapter.wait_board_change(last_hash, timeout_s=args.timeout)
            if not new_grid:
                # IMPROVEMENT: Handle case where click didn't register
                if args.mode in ("friend-create", "friend-join"):
                    log("No board change; waiting longer for friend mode...")
                    new_grid, last_hash = adapter.wait_board_change(last_hash, timeout_s=int(args.friend_wait_timeout))
                    if not new_grid:
                        log("Friend did not join/start within timeout; stopping.")
                        break
                else:
                    log("AI move did not register; stopping.")
                    break

            grid = new_grid
            cur_player *= -1  # Toggle to opponent's turn
            board_np = np.array(adapter.grid_to_board(grid), dtype=int)

            # Check if AI won
            if report_game_end(g, board_np, cur_player, last_mover="AI"):
                break

            # Wait for opponent's move
            new_grid, last_hash = adapter.wait_board_change(last_hash, timeout_s=args.timeout)
            if new_grid:
                grid = new_grid
                cur_player *= -1  # Back to AI's turn
                board_np = np.array(adapter.grid_to_board(grid), dtype=int)
                if report_game_end(g, board_np, cur_player, last_mover="Opponent"):
                    break
                move_idx += 1
            else:
                log("Opponent did not move within timeout.")
                break

        browser.close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
