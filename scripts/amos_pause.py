"""
Pause / resume the nnU-Net queue without losing progress (the cascade study: scripts/cascade_queue.py; the earlier
AMOS queue scripts/amos_nnunet_queue.py is also recognised as running).

    python scripts/amos_pause.py pause    # the run stops cleanly after the epoch it is in (a few minutes at most)
    python scripts/amos_pause.py resume   # starts the queue again in the background; it continues from the last epoch
    python scripts/amos_pause.py status   # running or not, current run and epoch

Every epoch writes checkpoint_latest.pth (weights, optimizer, LR schedule, logs); resuming loads it (nnUNetv2_train --c).
Even a hard kill or a restart of the laptop loses at most the epoch in progress when the queue is started again.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from nnunet_ext.nn import DEFAULT_ROOT  # noqa: E402

PAUSE = os.path.join(DEFAULT_ROOT, "PAUSE")


def running() -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'amos_nnunet_queue|cascade_queue' }).Count"],
                         capture_output=True, text=True).stdout.strip()
    return out not in ("", "0")


def status():
    print("queue running:", running(), "| pause requested:", os.path.exists(PAUSE))
    logs = sorted(glob.glob(os.path.join(DEFAULT_ROOT, "nnUNet_results", "Dataset0*", "*", "fold_*", "training_log_*.txt")),
                  key=os.path.getmtime)
    if logs:
        last = logs[-1]
        eps = re.findall(r"Epoch (\d+) *\n", open(last, encoding="utf-8", errors="replace").read())
        print("latest log:", os.path.relpath(last, DEFAULT_ROOT), "| epoch", eps[-1] if eps else "-")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "pause":
        open(PAUSE, "w").close()
        print("pause requested: the run stops after its current epoch")
    elif cmd == "resume":
        if running():
            raise SystemExit("the queue is already running")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        log = open(os.path.join(REPO, "logs", "cascade_queue.out"), "a")
        subprocess.Popen([sys.executable, "-W", "ignore", os.path.join(REPO, "scripts", "cascade_queue.py")], cwd=REPO, env=env,
                         stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000 | 0x00000200)  # hidden console, own process group
        print("queue started (it resumes from the last saved epoch)")
    else:
        status()
