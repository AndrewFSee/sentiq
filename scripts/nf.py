"""Run `newsfeed` (or any venv module / script) with the *base* Python and the venv's packages.

    set PY=%APPDATA%\\uv\\python\\cpython-3.11.12-windows-x86_64-none\\python.exe
    "%PY%" scripts\\nf.py collect --once           # = python -m newsfeed collect --once
    "%PY%" scripts\\nf.py -m pytest -q             # any module installed in the venv
    "%PY%" scripts\\nf.py scripts\\distill_train.py --name X ...   # a script

Why: Windows Smart App Control (enforcing on this laptop) blocks the venv's own executables,
which uv generates per environment and which therefore have no reputation: first
`.venv\\Scripts\\newsfeed.exe` (2026-10-24), then `.venv\\Scripts\\python.exe` (2026-10-26).
The base interpreter uv installed is allowed. This adds the venv's site-packages (incl. the
editable install of `newsfeed`) to the base interpreter and runs the CLI like
`python -m newsfeed`.
"""

import os
import runpy
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / ".venv" / "Lib" / "site-packages"
site.addsitedir(str(SITE))  # also runs its .pth files (the editable install of newsfeed)
# child processes started with sys.executable (= the base interpreter) inherit the same packages
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(SITE), str(ROOT), os.environ.get("PYTHONPATH", "")) if p
)
if sys.argv[1:2] == ["-m"]:  # any module from the venv: nf.py -m pytest -q / nf.py -m ruff check .
    module, args = sys.argv[2], sys.argv[3:]
elif sys.argv[1:2] and sys.argv[1].endswith(".py"):  # a script: nf.py scripts/distill_train.py ...
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
    sys.exit(0)
else:
    module, args = "newsfeed", sys.argv[1:]
sys.argv = [module, *args]
runpy.run_module(module, run_name="__main__", alter_sys=True)
