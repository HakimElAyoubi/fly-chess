"""Label positions with Stockfish's choice (Phase 3b).

    python -m flybrain.label data/positions_pilot.jsonl.gz [--depth 10] [--workers 8]

The fly was trained to imitate human moves, but a human plays Stockfish's best move only about
43% of the time, so more than half of that training signal was noise. This relabels the same
positions with an engine's choice at a fixed depth, and records the evaluation so the value head
can be trained on a position's real worth instead of how the game happened to end.

Adds three fields to each record and writes <name>_sf.jsonl.gz:
  sf_move  the engine's best move, uci
  sf_cp    evaluation in centipawns from the side to move's point of view (None if forced mate)
  sf_mate  moves to mate from the side to move's point of view, negative if being mated
Progress in data/progress.txt.
"""
import argparse
import gzip
import json
import multiprocessing as mp
import time
import chess
import chess.engine
from .graph import DATA
from .progress import write as progress

_engine = None


def _init(binary, depth, hash_mb):
    global _engine, _depth
    _engine = chess.engine.SimpleEngine.popen_uci(binary)
    _engine.configure({"Threads": 1, "Hash": hash_mb})
    _depth = depth


def _label(line):
    r = json.loads(line)
    board = chess.Board(r["fen"])
    if board.is_game_over():
        return None
    try:
        info = _engine.analyse(board, chess.engine.Limit(depth=_depth))
    except chess.engine.EngineError:
        return None
    score = info["score"].relative
    r["sf_move"] = info["pv"][0].uci()
    r["sf_cp"] = score.score()                      # None when the line is a forced mate
    r["sf_mate"] = score.mate()
    return json.dumps(r)


def run(path, depth=10, workers=8, binary="stockfish", hash_mb=64, chunk=64):
    out_path = str(path).replace(".jsonl.gz", "_sf.jsonl.gz")
    lines = [l for l in gzip.open(path, "rt")]
    n = len(lines)
    print(f"labelling {n:,} positions at depth {depth} with {workers} workers -> {out_path}", flush=True)
    t0 = time.time()
    agree = kept = 0
    with mp.Pool(workers, initializer=_init, initargs=(binary, depth, hash_mb)) as pool, gzip.open(out_path, "wt") as f:
        for i, res in enumerate(pool.imap(_label, lines, chunksize=chunk), 1):
            if res is not None:
                f.write(res + "\n"); kept += 1
                r = json.loads(res)
                agree += r["sf_move"] == r["move"]
            if i % 500 == 0 or i == n:
                progress(i, n, t0, f"stockfish labels d{depth}")
    dt = time.time() - t0
    print(f"wrote {kept:,} labelled positions in {dt/60:.1f} min ({kept/dt:.0f}/s)", flush=True)
    print(f"the human played the engine's choice in {agree/max(kept,1):.1%} of them", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("positions")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--hash", type=int, default=64)
    a = ap.parse_args()
    run(a.positions, a.depth, a.workers, a.stockfish, a.hash)
