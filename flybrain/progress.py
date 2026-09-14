"""Progress bar for long jobs. Every long-running step writes data/progress.txt so the user can
check it at any time with `cat data/progress.txt` (or `watch -n 5 cat data/progress.txt`)."""
import time
from pathlib import Path

PROGRESS = Path(__file__).resolve().parent.parent / "data" / "progress.txt"


def bar(done, total, t0, label="", width=30):
    frac = done / total if total else 1.0
    filled = int(round(frac * width))
    elapsed = time.time() - t0
    eta = (elapsed / frac - elapsed) if frac > 0 else float("nan")
    return f"{label:24s} [{'█' * filled}{'░' * (width - filled)}] {frac:6.1%} · {done}/{total} · {elapsed/60:.1f} min elapsed · {eta/60:.1f} min left"


def write(done, total, t0, label=""):
    line = bar(done, total, t0, label)
    PROGRESS.write_text(line + ("\n" if done < total else "\ndone\n"))
    return line
