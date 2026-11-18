"""The collector loop's single-instance lock (scripts/collect_service.py)."""

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_service.py"


def _load():
    spec = importlib.util.spec_from_file_location("collect_service", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_second_holder_is_refused_until_release(tmp_path):
    svc = _load()
    a, b = svc.InstanceLock(tmp_path / "c.lock"), svc.InstanceLock(tmp_path / "c.lock")
    assert a.acquire()
    assert not b.acquire()
    a.release()
    assert b.acquire()
    b.release()


def test_lock_held_by_another_process_and_freed_when_it_dies(tmp_path):
    svc = _load()
    path = tmp_path / "c.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys, time\n"
            f"spec = importlib.util.spec_from_file_location('s', r'{SCRIPT}')\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            f"lk = m.InstanceLock(__import__('pathlib').Path(r'{path}')); assert lk.acquire()\n"
            "print('held', flush=True); time.sleep(60)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        assert not svc.InstanceLock(path).acquire()
    finally:
        holder.kill()  # no release: the OS must drop the lock with the process
        holder.wait()
    # Windows drops a dead process's locks asynchronously; the standby loop retries anyway
    lock = svc.InstanceLock(path)
    deadline = time.monotonic() + 10
    while not lock.acquire():
        assert time.monotonic() < deadline, "lock not freed after the holder died"
        time.sleep(0.1)
    lock.release()
