# play_online_with_human.py
import re
import time
import numpy as np
from playwright.sync_api import sync_playwright
from alpha-zero-general.connect4.Connect4Game import Connect4Game
from alpha-zero-general.connect4.pytorch.NNet import NNetWrapper as NNet
from alpha-zero-general.MCTS import MCTS
from alpha-zero-general.utils import dotdict

CELL_SEL = ".grid-item"
CELL_CLASS_RE = re.compile(r"\bcell-(\d+)-(\d+)\b")  # 1-based row, col

def parse_board(page):
    # returns (board np.array (6x7), color_map dict(fill->+1/-1))
    cells = page.locator(CELL_SEL)
    count = cells.count()
    pieces = []
    fills = set()
    # Build temp grid as color strings or None
    grid = [[None for _ in range(7)] for _ in range(6)]
    for i in range(count):
        loc = cells.nth(i)
        cls = loc.get_attribute("class") or ""
        m = CELL_CLASS_RE.search(cls)
        if not m:
            continue
        R = int(m.group(1)) - 1
        C = int(m.group(2)) - 1
        # Try to locate a filled circle inside the SVG
        circ = loc.locator("svg circle:not(.empty-slot)")
        if circ.count() == 0:
            continue
        # Fallbacks for fill:
        fill = circ.first.get_attribute("fill")
        if not fill:
            # sometimes computed style stores fill
            fill = circ.first.evaluate("el => getComputedStyle(el).fill")
        if fill:
            fills.add(fill)
            grid[R][C] = fill

    return grid, fills

def grid_to_board(grid, my_fill, opp_fill):
    board = np.zeros((6,7), dtype=int)
    for r in range(6):
        for c in range(7):
            f = grid[r][c]
            if f is None:
                continue
            if f == my_fill:
                board[r, c] = 1
            elif f == opp_fill:
                board[r, c] = -1
    return board

def choose_ai_move(g, mcts, board_np, cur_player):
    canonical = g.getCanonicalForm(board_np, cur_player)
    pi = mcts.getActionProb(canonical, temp=0)
    return int(np.argmax(pi))

def click_column(page, col):
    # Click top cell of the column
    sel = f'.grid-item.cell-1-{col+1} > .position-relative > svg'
    page.locator(sel).click()

def wait_board_change(page, last_hash, timeout=20.0):
    start = time.time()
    while time.time() - start < timeout:
        grid, _ = parse_board(page)
        h = hash(str(grid))
        if h != last_hash:
            return grid, h
        time.sleep(0.3)
    return None, last_hash

def main_game(url, checkpoint_folder="./temp/", checkpoint_file="best.pth.tar", cpuct=1.0, sims=100):
    g = Connect4Game()
    nnet = NNet(g); nnet.load_checkpoint(checkpoint_folder, checkpoint_file)
    mcts = MCTS(g, nnet, dotdict({'numMCTSSims': sims, 'cpuct': float(cpuct)}))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")

        # Wait for the grid to be present
        page.wait_for_selector(CELL_SEL)

        # Infer colors after first move if needed
        grid, fills = parse_board(page)
        my_fill = opp_fill = None
        last_hash = hash(str(grid))

        # Loop
        cur_player = 1  # assume we play as canonical +1; if site assigns colors arbitrarily, we fix after first move
        while True:
            # If colors unresolved and it's our move, play a safe column to learn our color
            if my_fill is None:
                # find first valid col from internal perspective
                tmp_board = grid_to_board(grid, my_fill="", opp_fill="")  # empty => all None => zeros
                valids = g.getValidMoves(tmp_board, cur_player)
                col = int(np.where(valids)[0][0])
                click_column(page, col)
                # wait for board change; our new piece color is the one that increased
                new_grid, last_hash = wait_board_change(page, last_hash, timeout=10) or (grid, last_hash)
                if new_grid:
                    before_colors = sum(v is not None for row in grid for v in row)
                    after_colors = sum(v is not None for row in new_grid for v in row)
                    # detect newly placed cell
                    added = []
                    for r in range(6):
                        for c in range(7):
                            if grid[r][c] is None and new_grid[r][c] is not None:
                                added.append(new_grid[r][c])
                    if added:
                        my_fill = added[-1]
                        # define opp_fill as any other seen fill later
                    grid = new_grid
                    continue

            # Build numeric board (if opp_fill unknown, treat unknown as -1 once seen)
            # Once both fills are known, mapping is stable
            if my_fill and not opp_fill:
                # find any filled cell != my_fill
                for r in range(6):
                    for c in range(7):
                        f = grid[r][c]
                        if f and f != my_fill:
                            opp_fill = f
                            break

            board_np = grid_to_board(grid, my_fill or "MY", opp_fill or "OPP")
            # Decide on move if it is our turn (heuristic: try a click and see if board changes; or check if column overlays exist)
            # Safer: pick move and click; if not our turn, click will do nothing; we then wait for opponent change
            valids = g.getValidMoves(g.getCanonicalForm(board_np, cur_player), 1)
            if not valids.any():
                break

            col = choose_ai_move(g, mcts, board_np, cur_player)
            click_column(page, col)

            # Wait for our move to register
            new_grid, last_hash = wait_board_change(page, last_hash, timeout=10)
            if new_grid:
                grid = new_grid
            else:
                # maybe it wasn't our turn; wait for opponent
                new_grid, last_hash = wait_board_change(page, last_hash, timeout=30)
                if new_grid:
                    grid = new_grid
                    continue
                else:
                    print("No board change detected; stopping.")
                    break

            # Wait for opponent move
            new_grid, last_hash = wait_board_change(page, last_hash, timeout=30)
            if new_grid:
                grid = new_grid
            else:
                print("Opponent did not move within timeout.")
                break

        browser.close()