"""`python -m newsfeed ...` = the `newsfeed` command.

Windows Smart App Control (enforced on this laptop since 2026-10-24) blocks the unsigned
`.venv\\Scripts\\newsfeed.exe` launcher that uv generates; the signed python.exe is allowed, so
scripts call `uv run python -m newsfeed` instead of `uv run newsfeed`.
"""

from newsfeed.cli import app

if __name__ == "__main__":
    app(prog_name="newsfeed")
