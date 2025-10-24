from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=100)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        page.goto("https://papergames.io")
        page.pause()  # Opens Playwright Inspector so you can explore/select
        browser.close()

if __name__ == "__main__":
    main()
