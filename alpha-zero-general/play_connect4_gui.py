import sys
import argparse
import pygame
import numpy as np

from utils import dotdict
from connect4.Connect4Game import Connect4Game
from connect4.pytorch.NNet import NNetWrapper as NNet
from MCTS import MCTS

# Visual settings
CELL_SIZE = 90
PADDING = 20
PIECE_RADIUS = CELL_SIZE // 2 - 6
BG_COLOR = (20, 24, 35)
GRID_COLOR = (40, 80, 160)
EMPTY_COLOR = (230, 230, 230)
P1_COLOR = (235, 64, 52)   # Red
P2_COLOR = (245, 202, 52)  # Yellow
TEXT_COLOR = (230, 230, 230)


def draw_board(screen, game, board):
    h, w = game.getBoardSize()
    # Draw grid background
    board_rect = pygame.Rect(PADDING, PADDING + 60, w * CELL_SIZE, h * CELL_SIZE)
    pygame.draw.rect(screen, GRID_COLOR, board_rect, border_radius=10)

    # Draw cells (note: numpy row 0 is top; visually show row 0 at top to match prints)
    for r in range(h):
        for c in range(w):
            cx = PADDING + c * CELL_SIZE + CELL_SIZE // 2
            cy = PADDING + 60 + r * CELL_SIZE + CELL_SIZE // 2
            val = board[r][c]
            if val == 0:
                color = EMPTY_COLOR
            elif val == 1:
                color = P1_COLOR
            else:
                color = P2_COLOR
            pygame.draw.circle(screen, color, (cx, cy), PIECE_RADIUS)


def col_from_mouse(game, pos):
    _, w = game.getBoardSize()
    x, y = pos
    x -= PADDING
    y -= (PADDING + 60)
    if x < 0 or y < 0:
        return None
    c = x // CELL_SIZE
    if 0 <= c < w:
        return int(c)
    return None


def info_text(screen, font, msg):
    surf = font.render(msg, True, TEXT_COLOR)
    screen.blit(surf, (PADDING, PADDING))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--win-length", type=int, default=4)
    parser.add_argument("--checkpoint-folder", type=str, default="./temp/")
    parser.add_argument("--checkpoint-file", type=str, default="best.pth.tar")
    parser.add_argument("--num-mcts-sims", type=int, default=100)
    parser.add_argument("--cpuct", type=float, default=1.0)
    parser.add_argument("--human-first", action="store_true")
    args = parser.parse_args()

    g = Connect4Game(height=args.height, width=args.width, win_length=args.win_length)

    # Load AI net + MCTS
    nnet = NNet(g)
    nnet.load_checkpoint(args.checkpoint_folder, args.checkpoint_file)
    mcts_args = dotdict({'numMCTSSims': int(args.num_mcts_sims), 'cpuct': float(args.cpuct)})
    mcts = MCTS(g, nnet, mcts_args)

    pygame.init()
    pygame.display.set_caption("RL- human vs agent Connect4 - GUI")

    h, w = g.getBoardSize()
    screen_w = PADDING * 2 + w * CELL_SIZE
    screen_h = PADDING * 2 + 60 + h * CELL_SIZE
    screen = pygame.display.set_mode((screen_w, screen_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 24)

    board = g.getInitBoard()
    cur_player = 1 if args.human_first else -1  # human=1 if human-first, else AI moves first (AI is -1 here to flip to 1 in canonical)

    running = True
    game_over = False
    status = "Your turn" if cur_player == 1 else "AI thinking..."

    def ai_move(current_board, current_player):
        canonical = g.getCanonicalForm(current_board, current_player)
        pi = mcts.getActionProb(canonical, temp=0)
        action = int(np.argmax(pi))
        return g.getNextState(current_board, current_player, action)

    while running:
        screen.fill(BG_COLOR)

        # Header text
        if not game_over:
            msg = status + "  |  R = Restart, ESC = Quit"
        else:
            msg = status + "  |  R = Restart, ESC = Quit"
        info_text(screen, font, msg)

        draw_board(screen, g, board)
        pygame.display.flip()

        if not game_over and cur_player == -1:
            # AI move
            status = "AI thinking..."
            pygame.display.flip()
            pygame.event.pump()
            board, cur_player = ai_move(board, cur_player)
            res = g.getGameEnded(board, cur_player)
            if res != 0:
                game_over = True
                # After AI move, cur_player == human (1). res > 0 => human win, res < 0 => AI win.
                if res > 0:
                    status = "You win!"
                elif res < 0:
                    status = "AI wins!"
                else:
                    status = "Draw!"
            else:
                status = "Your turn"
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Restart game
                    board = g.getInitBoard()
                    cur_player = 1 if args.human_first else -1
                    game_over = False
                    status = "Your turn" if cur_player == 1 else "AI thinking..."
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not game_over:
                if cur_player == 1:
                    col = col_from_mouse(g, event.pos)
                    if col is None:
                        continue
                    # Validate move in canonical form (player 1 perspective)
                    valids = g.getValidMoves(g.getCanonicalForm(board, cur_player), 1)
                    if col < 0 or col >= len(valids) or not valids[col]:
                        continue
                    board, cur_player = g.getNextState(board, cur_player, col)
                    res = g.getGameEnded(board, cur_player)
                    if res != 0:
                        game_over = True
                        # After human move, cur_player == AI (-1). res > 0 => AI win, res < 0 => human win.
                        if res > 0:
                            status = "AI wins!"
                        elif res < 0:
                            status = "You win!"
                        else:
                            status = "Draw!"
                    else:
                        status = "AI thinking..."

        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
