from dataclasses import dataclass
from typing import Optional
from random import choice
from playwright.sync_api import sync_playwright, Locator

# --- Agent stub ---
@dataclass
class Agent:
    def choose_action(self, board=None) -> int:
        # TODO: plug in your RL policy later; random valid-looking column for now
        return choice(list(range(7)))

# --- Helpers ---
def safe_click_column(board_el: Locator, col: int, rows: int = 6, cols: int = 7):
    """
    Click at the horizontal center of the given column inside the board element.
    We compute a relative x,y so it works across screen sizes.
    """
    box = board_el.bounding_box()
    assert box is not None, "Board bounding box not found."
    x0, y0, w, h = box["x"], box["y"], box["width"], box["height"]
    col_width = w / cols
    # Click near the top of the board within that column (safe for gravity-based drop)
    click_x = x0 + (col + 0.5) * col_width
    click_y = y0 + h * 0.1
    board_el.page.mouse.click(click_x, click_y)

def main():
    agent = Agent()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=120)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        page.goto("https://papergames.io")

        # TODO: Use your recorded steps (from codegen) to navigate to Connect 4 and start a match vs bot/friend.
        # Example pattern (replace with real selectors you recorded):
        # page.get_by_role("link", name="Connect 4").click()
        # page.get_by_role("button", name="Play vs Computer").click()
        page.get_by_role("link", name="Connect 4 Connect").click()
        page.get_by_role("button", name="Play vs robot").click()
        page.get_by_role("textbox", name="Nickname").click()
        page.get_by_role("textbox", name="Nickname").fill("test")
        page.get_by_role("button", name="Continue").click()
        page.locator(".grid-item.cell-5-4 > .position-relative > svg > .empty-slot").click()
        page.locator(".grid-item.cell-5-2 > .position-relative > svg > .empty-slot").click()
        page.locator(".position-relative > svg > .circle-light").first.click()
        page.locator(".grid-item.cell-6-7 > .position-relative > svg > .empty-slot").click()
        page.locator(".grid-item.cell-6-1 > .position-relative > svg > .empty-slot").click()
        page.locator(".position-relative > svg > .circle-light").first.click()


        # --- Locate the board element ---
        # Start simple: many HTML5 games use <canvas>. Adjust this locator using codegen/inspector for the real board node.
        board = page.locator("canvas").first
        board.wait_for(state="visible", timeout=10000)

        # --- Wire agent -> click ---
        chosen_col = agent.choose_action(board=None)
        safe_click_column(board, chosen_col)

        page.wait_for_timeout(1500)  # brief pause to observe
        browser.close()

if __name__ == "__main__":
    main()
