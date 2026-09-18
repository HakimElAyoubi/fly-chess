"""Chess positions for Phase 3 from the Lichess open database.

Streams the monthly PGN (zstd-compressed, tens of GB) and stops after the requested number of
games, so only the first few hundred MB are downloaded. Keeps rated standard games where both
players are in a rating band, samples a few positions per game (skipping the opening plies),
and writes data/positions_<tag>.jsonl.gz with one record per position:
  {"fen", "move" (uci), "result" (1 white, 0 draw, -1 black), "elo", "game"}
Progress in data/progress.txt.
"""
import gzip
import io
import json
import sys
import time
import requests
import numpy as np
import zstandard
import chess.pgn
from .graph import DATA
from .progress import write as progress

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def stream(month="2026-08", n_games=40000, elo=(1600, 2200), per_game=8, skip_plies=6, tag="pilot", seed=0):
    rng = np.random.default_rng(seed)
    out_path = DATA / f"positions_{tag}.jsonl.gz"
    resp = requests.get(URL.format(month=month), headers={"User-Agent": "fly-chess/0.1"}, stream=True, timeout=60)
    resp.raise_for_status()
    reader = zstandard.ZstdDecompressor().stream_reader(resp.raw)
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    kept = seen = n_pos = 0
    t0 = time.time()
    with gzip.open(out_path, "wt") as f:
        while kept < n_games:
            game = chess.pgn.read_game(text)
            if game is None:
                break
            seen += 1
            h = game.headers
            try:
                w, b = int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))
            except ValueError:
                continue
            if not (elo[0] <= w <= elo[1] and elo[0] <= b <= elo[1]):
                continue
            if h.get("Variant", "Standard") != "Standard" or "Bullet" in h.get("Event", ""):
                continue
            res = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}.get(h.get("Result"))
            if res is None:
                continue
            moves = list(game.mainline_moves())
            if len(moves) <= skip_plies + 2:
                continue
            picks = sorted(rng.choice(np.arange(skip_plies, len(moves)), size=min(per_game, len(moves) - skip_plies), replace=False))
            board = game.board()
            for ply, mv in enumerate(moves):
                if ply in picks:
                    f.write(json.dumps({"fen": board.fen(), "move": mv.uci(), "result": res, "elo": (w + b) // 2, "game": kept + seed * 10_000_000}) + "\n")
                    n_pos += 1
                board.push(mv)
            kept += 1
            if kept % 200 == 0:
                progress(kept, n_games, t0, "lichess: streaming games")
    progress(n_games, n_games, t0, "lichess: streaming games")
    print(f"{out_path}: {n_pos} positions from {kept} games ({seen} scanned), {time.time() - t0:.0f}s")


if __name__ == "__main__":
    stream(n_games=int(sys.argv[1]) if len(sys.argv) > 1 else 40000, tag=sys.argv[2] if len(sys.argv) > 2 else "pilot",
           month=sys.argv[3] if len(sys.argv) > 3 else "2026-08", seed=int(sys.argv[4]) if len(sys.argv) > 4 else 0)
