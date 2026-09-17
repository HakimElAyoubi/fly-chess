"""Phase 4 matches: the fly against a random mover or against Stockfish at limited strength,
many games at once so the fly's moves are batched through the network.

    python -m flybrain.play --opponent random --games 200
    python -m flybrain.play --opponent stockfish --elo 1320 --games 200

Each game opens with four random plies for variety, then the fly (argmax over legal moves)
and the opponent alternate; colours alternate between games. A game ends by checkmate,
stalemate, insufficient material, a claimable draw (50-move or threefold repetition) or after
300 plies (draw). Writes data/match_<tag>.json (results, Elo with a 95% interval) and
data/match_<tag>.pgn. Progress in data/progress.txt.
"""
import argparse
import json
import math
import time
import numpy as np
import chess
import chess.engine
import chess.pgn
from .graph import DATA
from .uci import FlyEngine
from .progress import write as progress


def elo_from_score(s):
    s = min(max(s, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / s - 1.0)


def run(a):
    rng = np.random.default_rng(a.seed)
    fly = FlyEngine(a.weights, a.temperature, avoid_repetition=not a.allow_repetition)
    sf = None
    if a.opponent == "stockfish":
        sf = chess.engine.SimpleEngine.popen_uci(a.stockfish)
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": a.elo, "Threads": 1} if not a.sf_depth else {"Threads": 1})
    games = []
    t0 = time.time(); done = 0
    results = []            # (fly_result: 1 win / 0.5 draw / 0 loss, termination, plies, fly_color)
    pgn_out = open(DATA / f"match_{a.tag}.pgn", "w")
    for block in range(0, a.games, a.concurrency):
        active = []
        for g in range(block, min(block + a.concurrency, a.games)):
            b = chess.Board()
            for _ in range(a.opening_plies):
                moves = list(b.legal_moves)
                if not moves: break
                b.push(moves[int(rng.integers(len(moves)))])
            active.append({"id": g, "board": b, "fly_color": chess.WHITE if g % 2 == 0 else chess.BLACK})
        while active:
            fly_turn = [x for x in active if x["board"].turn == x["fly_color"] and not x["board"].is_game_over(claim_draw=True) and len(x["board"].move_stack) < a.max_plies]
            if fly_turn:
                moves, _ = fly.choose([x["board"] for x in fly_turn])
                for x, m in zip(fly_turn, moves):
                    x["board"].push(m)
            for x in active:
                b = x["board"]
                if b.turn != x["fly_color"] and not b.is_game_over(claim_draw=True) and len(b.move_stack) < a.max_plies:
                    if sf is None:
                        moves = list(b.legal_moves); b.push(moves[int(rng.integers(len(moves)))])
                    else:
                        b.push(sf.play(b, chess.engine.Limit(depth=a.sf_depth) if a.sf_depth else chess.engine.Limit(time=a.sf_time)).move)
            still = []
            for x in active:
                b = x["board"]
                over = b.is_game_over(claim_draw=True) or len(b.move_stack) >= a.max_plies
                if not over:
                    still.append(x); continue
                if b.is_checkmate():
                    res = 1.0 if b.turn != x["fly_color"] else 0.0; term = "checkmate"
                else:
                    res = 0.5
                    term = ("stalemate" if b.is_stalemate() else "insufficient material" if b.is_insufficient_material()
                            else "repetition" if b.can_claim_threefold_repetition() else "50-move rule" if b.can_claim_fifty_moves()
                            else "max plies")
                results.append((res, term, len(b.move_stack), "white" if x["fly_color"] else "black"))
                game = chess.pgn.Game.from_board(b)
                game.headers["White"] = "Fly" if x["fly_color"] else a.opponent; game.headers["Black"] = a.opponent if x["fly_color"] else "Fly"
                game.headers["Result"] = "1-0" if (res == 1.0) == (x["fly_color"] == chess.WHITE) and res != 0.5 else "0-1" if res != 0.5 else "1/2-1/2"
                game.headers["Round"] = str(x["id"] + 1); game.headers["Termination"] = term
                print(game, file=pgn_out, end="\n\n"); pgn_out.flush()
                done += 1
            active = still
            progress(done, a.games, t0, f"match vs {a.opponent}{' ' + str(a.elo) if sf else ''}")
    pgn_out.close()
    if sf: sf.quit()
    r = np.array([x[0] for x in results]); n = len(r)
    score = float(r.mean()); se = float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    lo, hi = max(0.0, score - 1.96 * se), min(1.0, score + 1.96 * se)
    summary = {
        "opponent": a.opponent + ((f" (depth {a.sf_depth})" if a.sf_depth else f" (UCI_Elo {a.elo})") if sf else ""), "games": n, "fly_temperature": a.temperature, "avoid_repetition": not a.allow_repetition,
        "wins": int((r == 1).sum()), "draws": int((r == 0.5).sum()), "losses": int((r == 0).sum()),
        "score": score, "score_95ci": [lo, hi],
        "elo_diff": elo_from_score(score), "elo_diff_95ci": [elo_from_score(lo), elo_from_score(hi)],
        "fly_elo_estimate": (a.elo + elo_from_score(score)) if (sf and not a.sf_depth) else None,
        "fly_elo_95ci": [a.elo + elo_from_score(lo), a.elo + elo_from_score(hi)] if (sf and not a.sf_depth) else None,
        "mean_plies": float(np.mean([x[2] for x in results])),
        "terminations": {t: int(sum(1 for x in results if x[1] == t)) for t in sorted(set(x[1] for x in results))},
        "score_as_white": float(np.mean([x[0] for x in results if x[3] == "white"])), "score_as_black": float(np.mean([x[0] for x in results if x[3] == "black"])),
        "minutes": (time.time() - t0) / 60,
    }
    json.dump(summary, open(DATA / f"match_{a.tag}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", choices=["random", "stockfish"], default="random")
    ap.add_argument("--games", type=int, default=200); ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--elo", type=int, default=1320); ap.add_argument("--sf-time", type=float, default=0.05); ap.add_argument("--sf-depth", type=int, default=0)
    ap.add_argument("--stockfish", default="stockfish"); ap.add_argument("--weights", default=str(DATA / "train_gpu" / "model_step11600.pt"))
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None); ap.add_argument("--allow-repetition", action="store_true")
    a = ap.parse_args()
    a.tag = a.tag or (a.opponent if a.opponent == "random" else f"stockfish_depth{a.sf_depth}" if a.sf_depth else f"stockfish{a.elo}")
    run(a)
