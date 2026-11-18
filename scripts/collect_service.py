"""Live collector loop with a single-instance lock: one `newsfeed collect --once` cycle, a health
check, then a pause (`--interval`, default 120 s = collect.interval_seconds), forever.

Started two ways, both through the base interpreter + scripts/nf.py (see nf.py):
  * at boot, before anyone logs in, by the scheduled task \\Sentiq\\Collector
    (scripts/install_boot_collector.ps1), so a Windows Update restart costs minutes, not the
    hours until the next login;
  * in a labelled console window (scripts/collect_window.cmd, Startup folder).

Only the process holding `data/state/collector.lock` (an OS byte-range lock, released when the
process dies, however it dies) collects. Any other copy stands by, shows the active copy's
status from `data/state/collector-status.json`, and takes over within `--standby-poll`
seconds if the active one goes away. So a window opened at logon next to the boot task never
doubles the SEC request rate.

Each cycle is a fresh process (a crash doesn't stop collection); its output is appended to
data/collect.log, this loop's own events to data/collector-service.log.

  python scripts/collect_service.py [--mode window|boot] [--once]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data" / "state"
LOCK_PATH = STATE / "collector.lock"
STATUS_PATH = STATE / "collector-status.json"
CYCLE_LOG = ROOT / "data" / "collect.log"
SERVICE_LOG = ROOT / "data" / "collector-service.log"
LABEL = "sentiq - COLLECTING 8-Ks"


class InstanceLock:
    """Non-blocking exclusive lock on byte 0 of `path`; held until `release()` or process exit."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")  # noqa: SIM115 - held open for the lifetime of the lock
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"{now_iso()} pid {os.getpid()} {msg}"
    print(line, flush=True)
    try:
        with open(SERVICE_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def set_title(text: str) -> None:
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleTitleW(text)
        except Exception:  # no console (boot task): nothing to label
            pass


def write_status(status: dict) -> None:
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=1), encoding="utf-8")
    for _ in range(5):  # a standby reader may have the file open (Windows)
        try:
            os.replace(tmp, STATUS_PATH)
            return
        except PermissionError:
            time.sleep(0.2)


def read_status() -> dict:
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def nf(*args: str, log_to: Path | None) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / "nf.py"), *args]
    if log_to is None:
        return subprocess.call(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(log_to, "ab") as out:
        return subprocess.call(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)


def run_cycles(mode: str, interval: int, once: bool) -> None:
    status = {"pid": os.getpid(), "mode": mode, "started_at": now_iso(), "cycles": 0}
    while True:
        status["last_cycle_start"] = now_iso()
        write_status(status)
        print(f"[{datetime.now(UTC).astimezone():%Y-%m-%d %H:%M:%S}] cycle...", flush=True)
        if nf("collect", "--once", log_to=CYCLE_LOG) != 0:
            result = "failed"
        elif nf("collect", "--check", log_to=None) != 0:
            result = "unhealthy"
        else:
            result = "ok"
        status.update(cycles=status["cycles"] + 1, last_cycle_end=now_iso(), last_result=result)
        write_status(status)
        note = {
            "ok": "ok",
            "failed": "FAILED - see data\\collect.log",
            "unhealthy": "ok, but the health check reports a problem (collect --check)",
        }
        print(f"    {note[result]}", flush=True)
        if result != "ok":
            log(f"cycle {status['cycles']}: {result}")
        set_title(
            LABEL
            + {"ok": "", "failed": " [last cycle FAILED]", "unhealthy": " [UNHEALTHY]"}[result]
        )
        if once:
            return
        time.sleep(interval)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["window", "boot"], default="window")
    ap.add_argument("--interval", type=int, default=120, help="pause between cycles, seconds")
    ap.add_argument("--standby-poll", type=int, default=30, help="seconds between lock attempts")
    ap.add_argument("--once", action="store_true", help="one cycle, then exit (smoke test)")
    args = ap.parse_args()

    lock = InstanceLock(LOCK_PATH)
    shown = None
    while not lock.acquire():
        st = read_status()
        if args.once:
            print(
                f"Another collector is active (pid {st.get('pid')}, {st.get('mode')}); not running."
            )
            return 1
        view = (st.get("pid"), st.get("last_cycle_end"), st.get("last_result"))
        if view != shown:
            shown = view
            tag = {"failed": " [last cycle FAILED]", "unhealthy": " [UNHEALTHY]"}.get(
                st.get("last_result"), ""
            )
            set_title(f"sentiq - STANDBY, {st.get('mode', '?')} collector active{tag}")
            print(
                f"[{datetime.now(UTC).astimezone():%H:%M:%S}] standby: {st.get('mode', '?')} "
                f"collector pid {st.get('pid')} is active; last cycle {st.get('last_cycle_end')} "
                f"{st.get('last_result')}. Taking over if it stops.",
                flush=True,
            )
        time.sleep(args.standby_poll)

    log(f"collector started ({args.mode}), lock acquired")
    set_title(LABEL)
    try:
        run_cycles(args.mode, args.interval, args.once)
    finally:
        lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
