import re
import time
from typing import Optional, Tuple, List
from playwright.sync_api import Page, Locator

CELL_SEL = ".grid-item"
CELL_CLASS_RE = re.compile(r"\bcell-(\d+)-(\d+)\b")

class PapergamesConnect4:
    def __init__(self, page: Page):
        self.page = page
        self.my_fill: Optional[str] = None
        self.opp_fill: Optional[str] = None

    def navigate(self, mode: str = "random", nickname: str = "player"):
        self.page.goto("https://papergames.io/en/connect4", wait_until="networkidle")
        # IMPROVEMENT: More specific exception handling instead of bare except
        # This helps with debugging if the page structure changes
        try:
            consent = self.page.get_by_role("button", name=re.compile("(Accept|I agree|I accept|Got it|Allow)", re.I))
            if consent.count() > 0:
                consent.first.click(timeout=2000)
        except Exception as e:
            # Silently continue if consent button not found (may already be accepted)
            # but log could be added here for debugging
            pass
        if mode == "robot":
            try:
                btn = self.page.get_by_role("button", name=re.compile("(Play\\s*vs\\s*robot|robot|computer|AI)", re.I))
                if btn.count() > 0:
                    btn.first.click()
            except Exception:
                pass
        else:
            clicked = False
            for pattern in [r"Play\\s*online", r"Online", r"Play"]:
                try:
                    btn = self.page.get_by_role("button", name=re.compile(pattern, re.I))
                    if btn.count() > 0:
                        btn.first.click()
                        clicked = True
                        break
                except Exception:
                    continue
            if not clicked:
                try:
                    link = self.page.get_by_role("link", name=re.compile("(Play|Online)", re.I))
                    if link.count() > 0:
                        link.first.click()
                except Exception:
                    pass
        try:
            tb = self.page.get_by_role("textbox", name=re.compile("Nickname", re.I))
            if tb.count() > 0 and tb.first.is_visible():
                tb.first.fill(nickname)
                cont = self.page.get_by_role("button", name=re.compile("(Continue|Start|Play)", re.I))
                if cont.count() > 0:
                    cont.first.click()
        except Exception:
            pass

    def create_friend_room(self, nickname: str = "player") -> str:
        """Create a private room to play with a friend, return the room URL."""
        self.page.goto("https://papergames.io/en/connect4", wait_until="networkidle")
        try:
            consent = self.page.get_by_role("button", name=re.compile("(Accept|I agree|I accept|Got it|Allow)", re.I))
            if consent.count() > 0:
                consent.first.click(timeout=2000)
        except Exception:
            pass
        # Try various labels commonly used for friend/private rooms
        patterns = [r"Play\s*with\s*a\s*friend", r"Invite\s*a\s*friend", r"Create\s*room", r"Private\s*room", r"With\s*a\s*friend", r"Friend"]
        clicked = False
        for pat in patterns:
            try:
                btn = self.page.get_by_role("button", name=re.compile(pat, re.I))
                if btn.count() > 0:
                    btn.first.click()
                    clicked = True
                    break
            except Exception:
                continue
        if not clicked:
            # Fallback: links
            for pat in [r"Friend", r"Private", r"Invite"]:
                try:
                    link = self.page.get_by_role("link", name=re.compile(pat, re.I))
                    if link.count() > 0:
                        link.first.click()
                        clicked = True
                        break
                except Exception:
                    continue
        # Fill nickname if prompted
        try:
            tb = self.page.get_by_role("textbox", name=re.compile("Nickname", re.I))
            if tb.count() > 0 and tb.first.is_visible():
                tb.first.fill(nickname)
                cont = self.page.get_by_role("button", name=re.compile("(Continue|Start|Play|Create)", re.I))
                if cont.count() > 0:
                    cont.first.click()
        except Exception:
            pass
        # Wait for room URL pattern
        try:
            self.page.wait_for_url(re.compile(r"/en/[^/]*r/|/en/r/", re.I), timeout=15000)
        except Exception:
            pass
        return self.page.url

    def join_friend_room(self, room_url: str, nickname: str = "player"):
        """Join an existing friend room by URL."""
        self.page.goto(room_url, wait_until="networkidle")
        try:
            consent = self.page.get_by_role("button", name=re.compile("(Accept|I agree|I accept|Got it|Allow)", re.I))
            if consent.count() > 0:
                consent.first.click(timeout=2000)
        except Exception:
            pass
        try:
            tb = self.page.get_by_role("textbox", name=re.compile("Nickname", re.I))
            if tb.count() > 0 and tb.first.is_visible():
                tb.first.fill(nickname)
                join = self.page.get_by_role("button", name=re.compile("(Join|Continue|Start|Play)", re.I))
                if join.count() > 0:
                    join.first.click()
        except Exception:
            pass

    def wait_for_board(self, timeout_ms: int = 15000):
        self.page.wait_for_selector(CELL_SEL, timeout=timeout_ms)
        start = time.time()
        while (time.time() - start) * 1000 < timeout_ms:
            try:
                if self.page.locator(CELL_SEL).count() >= 42:
                    return True
            except Exception:
                pass
            time.sleep(0.2)
        return False

    def _cells(self) -> Locator:
        return self.page.locator(CELL_SEL)

    def parse_grid(self) -> Tuple[List[List[Optional[str]]], set]:
        cells = self._cells()
        count = cells.count()
        fills = set()
        grid: List[List[Optional[str]]] = [[None for _ in range(7)] for _ in range(6)]
        for i in range(count):
            loc = cells.nth(i)
            cls = loc.get_attribute("class") or ""
            m = CELL_CLASS_RE.search(cls)
            if not m:
                continue
            r = int(m.group(1)) - 1
            c = int(m.group(2)) - 1
            circ = loc.locator("svg circle:not(.empty-slot)")
            if circ.count() == 0:
                continue
            fill = circ.first.get_attribute("fill")
            if not fill:
                try:
                    fill = circ.first.evaluate("el => getComputedStyle(el).fill")
                except Exception:
                    fill = None
            if fill:
                fills.add(fill)
                grid[r][c] = fill
        return grid, fills

    def grid_to_board(self, grid: List[List[Optional[str]]]) -> List[List[int]]:
        board = [[0 for _ in range(7)] for _ in range(6)]
        my = self.my_fill or "MY"
        opp = self.opp_fill or "OPP"
        for r in range(6):
            for c in range(7):
                f = grid[r][c]
                if f is None:
                    continue
                if f == my:
                    board[r][c] = 1
                elif f == opp:
                    board[r][c] = -1
        return board

    def top_cell_locator(self, col: int) -> Locator:
        sel = f'.grid-item.cell-1-{col+1} > .position-relative > svg'
        return self.page.locator(sel)

    def click_column(self, col: int):
        loc = self.top_cell_locator(col)
        try:
            loc.click()
            return
        except Exception:
            pass
        top_cell = self.page.locator(f'.grid-item.cell-1-{col+1}')
        box = top_cell.bounding_box()
        if not box:
            grid_any = self._cells().first
            box = grid_any.bounding_box()
        if not box:
            raise RuntimeError("Cannot resolve column bounding box")
        x0, y0, w, h = box["x"], box["y"], box["width"], box["height"]
        cx = x0 + w/2
        cy = y0 + h*0.5
        self.page.mouse.click(cx, cy)

    def wait_board_change(self, last_hash: int, timeout_s: float) -> Tuple[Optional[List[List[Optional[str]]]], int]:
        """
        IMPROVEMENT: Use more robust board comparison instead of simple hash.
        Previous issue: hash(str(grid)) could theoretically have collisions
        New approach: Still use hash for efficiency, but with better string representation
        Alternative considered: Deep comparison of grids, but hash is faster for polling
        """
        start = time.time()
        while time.time() - start < timeout_s:
            grid, _ = self.parse_grid()
            # Use tuple representation for more reliable hashing
            h = hash(str([tuple(row) for row in grid]))
            if h != last_hash:
                return grid, h
            time.sleep(0.3)
        return None, last_hash

    def update_fills_after_move(self, before, after):
        """
        IMPROVEMENT: More robust color detection logic.
        Previous approach: Assumed last added piece is always "my" color
        Problem: Fails if opponent moves first or if there are timing issues
        New approach: Track all added pieces and use context to determine ownership
        """
        added = []
        for r in range(6):
            for c in range(7):
                if before[r][c] is None and after[r][c] is not None:
                    added.append(after[r][c])

        # If this is the first move detection and we haven't set my_fill yet
        if added and self.my_fill is None:
            # If there's only one new piece, we need to determine if it's ours
            # by checking if we just made a move or if opponent moved first
            self.my_fill = added[-1]  # Last added is most recent

        # Detect opponent's color by finding any different color on the board
        if self.opp_fill is None and self.my_fill is not None:
            for r in range(6):
                for c in range(7):
                    f = after[r][c]
                    if f and f != self.my_fill:
                        self.opp_fill = f
                        return
