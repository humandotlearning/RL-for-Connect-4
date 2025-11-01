import re
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://papergames.io/en/")
    page.get_by_role("link", name="Gomoku Gomoku").click()
    page.get_by_role("button", name="Play vs robot").click()
    page.get_by_role("textbox", name="Nickname").click()
    page.get_by_role("textbox", name="Nickname").fill("test")
    page.get_by_role("button", name="Continue").click()
    page.locator(".cell-7-7").click()
    page.locator(".cell-6-5").click()
    page.locator(".cell-12-2").click()
    page.get_by_role("button", name="Resign").click()
    page.get_by_role("button", name="Resign").click()

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
