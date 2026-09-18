"""Progress bars for long jobs.

Several jobs run at once on this machine, so each one owns a slot in data/progress.json rather
than fighting over a single file. The slot is taken from the first word of the label, so
"match vs random" and "stockfish labels d10" never overwrite each other. data/progress.txt is
kept as a plain-text dump of every active job for `cat`, and progress.html renders the JSON.
"""
import json
import os
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
PROGRESS = DATA / "progress.txt"
STATE = DATA / "progress.json"
STALE_SECONDS = 1800          # a slot nobody has touched for half an hour is dropped


def bar(done, total, t0, label="", width=30):
    frac = done / total if total else 1.0
    filled = int(round(frac * width))
    elapsed = time.time() - t0
    eta = (elapsed / frac - elapsed) if frac > 0 else float("nan")
    return f"{label:24s} [{'█' * filled}{'░' * (width - filled)}] {frac:6.1%} · {done}/{total} · {elapsed/60:.1f} min elapsed · {eta/60:.1f} min left"


def _slot(label):
    head = (label or "job").split(":")[0].split()[0]
    return "".join(c for c in head.lower() if c.isalnum()) or "job"


def write(done, total, t0, label="", slot=None):
    """Record this job's progress. Returns the one-line bar."""
    line = bar(done, total, t0, label)
    slot = slot or os.environ.get("FLY_PROGRESS_SLOT") or _slot(label)
    now = time.time()
    try:
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
    except (ValueError, OSError):
        state = {}
    state = {k: v for k, v in state.items() if now - v.get("ts", 0) < STALE_SECONDS}
    state[slot] = {"label": label, "done": done, "total": total, "elapsed_min": (now - t0) / 60,
                   "eta_min": None if not done else ((now - t0) / (done / total) - (now - t0)) / 60,
                   "pid": os.getpid(), "ts": now, "finished": done >= total, "line": line}
    try:                                      # write then rename, so a reader never sees half a file
        tmp = STATE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=1)); os.replace(tmp, STATE)
        PROGRESS.write_text("\n".join(v["line"] + ("\ndone" if v["finished"] else "")
                                      for v in sorted(state.values(), key=lambda v: -v["ts"])) + "\n")
    except OSError:
        pass
    return line
