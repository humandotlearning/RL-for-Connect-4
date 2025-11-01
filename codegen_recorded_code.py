import re
from playwright.sync_api import Playwright, sync_playwright, expect


def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://papergames.io/en/")
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

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
