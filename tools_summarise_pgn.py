"""Summarise a match PGN into the same shape flybrain/play.py writes.

    python tools_summarise_pgn.py data/match_sfd1_rl.pgn "Stockfish 17 (depth 1)"

Needed because one match was stopped with three of its two hundred games still unfinished: a
Stockfish call blocked on a long game. Scoring the completed games from the PGN is the honest
way to use the rest, and doing it for both arms with the same code keeps them comparable.
"""
import json, math, sys, chess.pgn

def elo_from_score(s):
    s = min(max(s, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / s - 1.0)

path, opponent = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "unknown")
res, plies, terms, colours = [], [], {}, []
with open(path) as f:
    while (g := chess.pgn.read_game(f)) is not None:
        h = g.headers
        fly_white = h["White"] == "Fly"
        r = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[h["Result"]]
        res.append(r if fly_white else 1.0 - r)
        colours.append("white" if fly_white else "black")
        terms[h.get("Termination", "?")] = terms.get(h.get("Termination", "?"), 0) + 1
        b = g.end().board(); plies.append(len(b.move_stack))
n = len(res); score = sum(res) / n
var = sum((x - score) ** 2 for x in res) / (n - 1)
se = math.sqrt(var / n)
lo, hi = max(0.0, score - 1.96 * se), min(1.0, score + 1.96 * se)
out = {"opponent": opponent, "games": n, "fly_temperature": 0.0, "avoid_repetition": True,
       "wins": sum(1 for x in res if x == 1), "draws": sum(1 for x in res if x == 0.5),
       "losses": sum(1 for x in res if x == 0), "score": score, "score_95ci": [lo, hi],
       "elo_diff": elo_from_score(score), "elo_diff_95ci": [elo_from_score(lo), elo_from_score(hi)],
       "fly_elo_estimate": None, "fly_elo_95ci": None,
       "mean_plies": sum(plies) / n, "terminations": terms,
       "score_as_white": sum(r for r, c in zip(res, colours) if c == "white") / max(1, colours.count("white")),
       "score_as_black": sum(r for r, c in zip(res, colours) if c == "black") / max(1, colours.count("black")),
       "from_pgn": path}
json.dump(out, open(path.replace(".pgn", ".json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("opponent", "games", "wins", "draws", "losses", "score", "score_95ci", "mean_plies")}, indent=1))
