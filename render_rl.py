"""Draw the Phase 7 learning curve and the baseline-versus-RL comparison.

    python render_rl.py [--tag ppo]

Reads data/rl_<tag>/log.jsonl (written by flybrain.rl: one row per iteration, plus one row per
evaluation) and the match files written by remote/after_rl.sh, and writes data/rl_curve.svg and
data/rl_results.json. Hand-rolled SVG, like the other figures here, so the repository needs no
plotting dependency.
"""
import argparse, glob, json, os

W, H = 980, 620
PAD_L, PAD_R, PAD_T = 70, 96, 34   # the right margin carries the second panel's KL axis


def axes(x0, y0, w, h, xmax, ymin, ymax, xlab, ylab, yticks, xticks):
    """Frame, ticks and labels for one panel; returns the two mapping functions."""
    sx = lambda v: x0 + (v / xmax) * w if xmax else x0
    sy = lambda v: y0 + h - (v - ymin) / (ymax - ymin) * h
    o = [f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#d8d4cc"/>']
    for t in yticks:
        o.append(f'<line x1="{x0}" y1="{sy(t):.1f}" x2="{x0+w}" y2="{sy(t):.1f}" stroke="#eeebe4"/>')
        o.append(f'<text x="{x0-8}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="11" fill="#6b6459">{t:g}</text>')
    for t in xticks:
        o.append(f'<line x1="{sx(t):.1f}" y1="{y0+h}" x2="{sx(t):.1f}" y2="{y0+h+4}" stroke="#9a9184"/>')
        o.append(f'<text x="{sx(t):.1f}" y="{y0+h+18}" text-anchor="middle" font-size="11" fill="#6b6459">{t:g}</text>')
    o.append(f'<text x="{x0+w/2:.0f}" y="{y0+h+36}" text-anchor="middle" font-size="12" fill="#3d382f">{xlab}</text>')
    o.append(f'<text x="{x0-52}" y="{y0+h/2:.0f}" text-anchor="middle" font-size="12" fill="#3d382f" '
             f'transform="rotate(-90 {x0-52} {y0+h/2:.0f})">{ylab}</text>')
    return sx, sy, o


def path(pts, colour, width=2, dash=None):
    d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    return f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>"


def nice_ticks(hi, n=5):
    step = max(1, int(round(hi / n / 10) * 10)) if hi > 40 else max(1, int(round(hi / n)))
    return [t for t in range(0, int(hi) + step, step) if t <= hi]      # never a tick past the axis


def render(tag, baseline_score, baseline_label):
    src = f"data/rl_{tag}/log.jsonl"
    if not os.path.exists(src):                       # the run is on a rented box; sync_rl.sh lands it here
        src = f"data/rl_{tag}/log_remote.jsonl"
    rows = [json.loads(l) for l in open(src)]
    train = [r for r in rows if r.get("kind") != "eval"]
    evals = [r for r in rows if r.get("kind") == "eval"]
    if not train:
        raise SystemExit("no iterations logged yet")
    xmax = max(r["iter"] for r in train)
    xt = nice_ticks(xmax)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="#fbfaf7"/>',
         f'<text x="{PAD_L}" y="22" font-size="15" fill="#241f18" font-family="Georgia,serif">'
         f'The fly learning from its own games</text>']

    # ---- panel 1: how it scores against the opponent it is training against
    w = W - PAD_L - PAD_R; h = 240; y0 = PAD_T + 12
    ys = [r["score_recent"] for r in train if r.get("score_recent") is not None] + [e["score"] for e in evals] + [baseline_score]
    lo, hi = min(ys + [0.0]), max(ys + [0.5])
    lo, hi = max(0.0, lo - 0.05), min(1.0, hi + 0.05)
    sx, sy, ax = axes(PAD_L, y0, w, h, xmax, lo, hi, "PPO iteration (each is 1,536 of the fly's own moves)",
                      "match score", [round(lo + i * (hi - lo) / 4, 2) for i in range(5)], xt)
    o += ax
    o.append(f'<line x1="{PAD_L}" y1="{sy(baseline_score):.1f}" x2="{PAD_L+w}" y2="{sy(baseline_score):.1f}" '
             f'stroke="#b0453a" stroke-width="1.6" stroke-dasharray="6 4"/>')
    o.append(f'<text x="{PAD_L+w-4}" y="{sy(baseline_score)-7:.1f}" text-anchor="end" font-size="11" fill="#b0453a">'
             f'{baseline_label} ({baseline_score:.3f})</text>')
    run = [(sx(r["iter"]), sy(r["score_recent"])) for r in train if r.get("score_recent") is not None]
    if run: o.append(path(run, "#9a9184", 1.2))
    if evals:
        o.append(path([(sx(e["iter"]), sy(e["score"])) for e in evals], "#2f6f4f", 2.4))
        for e in evals:
            o.append(f'<circle cx="{sx(e["iter"]):.1f}" cy="{sy(e["score"]):.1f}" r="3" fill="#2f6f4f"/>')
    o.append(f'<text x="{PAD_L+8}" y="{y0+16}" font-size="11" fill="#2f6f4f">held-out evaluation, argmax play</text>')
    o.append(f'<text x="{PAD_L+8}" y="{y0+31}" font-size="11" fill="#9a9184">running score of the games it is playing</text>')

    # ---- panel 2: is it still exploring, and how far has it drifted from what it was taught
    y1 = y0 + h + 62; h2 = 200
    emax = max([r["entropy"] for r in train] + [1.0]) * 1.1
    kmax = max([r["kl_ref"] for r in train] + [0.1]) * 1.1
    sx2, sy2, ax2 = axes(PAD_L, y1, w, h2, xmax, 0, emax, "PPO iteration",
                         "entropy of the move choice (nats)", [round(i * emax / 4, 2) for i in range(5)], xt)
    o += ax2
    o.append(path([(sx2(r["iter"]), sy2(r["entropy"])) for r in train], "#3f5fa8", 1.8))
    ky = lambda v: y1 + h2 - v / kmax * h2
    o.append(path([(sx2(r["iter"]), ky(r["kl_ref"])) for r in train], "#a8762f", 1.8))
    for i in range(5):
        v = i * kmax / 4
        o.append(f'<text x="{PAD_L+w+8}" y="{ky(v)+4:.1f}" font-size="11" fill="#a8762f">{v:.2f}</text>')
    o.append(f'<text x="{PAD_L+w+46}" y="{y1+h2/2:.0f}" text-anchor="middle" font-size="12" fill="#a8762f" '
             f'transform="rotate(90 {PAD_L+w+46} {y1+h2/2:.0f})">KL from the supervised policy</text>')
    o.append(f'<text x="{PAD_L+8}" y="{y1+16}" font-size="11" fill="#3f5fa8">exploration: how undecided its move choice is</text>')
    o.append(f'<text x="{PAD_L+8}" y="{y1+31}" font-size="11" fill="#a8762f">drift: how far it has moved from what it was taught</text>')
    o.append("</svg>")
    open("data/rl_curve.svg", "w").write("\n".join(o))
    return train, evals


def compare():
    """The matches replayed with both sets of weights under identical settings."""
    out = {}
    for f in sorted(glob.glob("data/match_*_baseline.json")) + sorted(glob.glob("data/match_*_rl.json")):
        tag = os.path.basename(f)[len("match_"):-len(".json")]
        m = json.load(open(f))
        out[tag] = {k: m[k] for k in ("opponent", "games", "wins", "draws", "losses", "score", "score_95ci", "mean_plies")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ppo")
    ap.add_argument("--baseline-score", type=float, default=0.2425)
    ap.add_argument("--baseline-label", default="supervised fly, Phase 4")
    a = ap.parse_args()
    train, evals = render(a.tag, a.baseline_score, a.baseline_label)
    res = {"iterations": len(train), "moves": len(train) * 1536,
           "best_eval": max((e["score"] for e in evals), default=None),
           "final_entropy": train[-1]["entropy"], "final_kl_ref": train[-1]["kl_ref"],
           "evals": [{"iter": e["iter"], "score": e["score"], "wins": e["wins"], "draws": e["draws"], "losses": e["losses"]} for e in evals],
           "matches": compare()}
    json.dump(res, open("data/rl_results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "evals"}, indent=1))
    print(f"wrote data/rl_curve.svg over {len(train)} iterations, {len(evals)} evaluations")
