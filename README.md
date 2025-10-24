

```
# 2) Add Playwright (Python version)
uv add playwright

# 3) Download browser binaries (Chromium is fine to start)
uv run python -m playwright install chromium
```

(Optional but useful) Recorder to learn selectors:

```
uv run playwright codegen https://papergames.io
# This opens a live browser. Click through to Connect 4; it records steps and shows Python code.
# Close when done; you can copy selectors from the right panel.
```
