"""Render the Phase 4 match results (data/match_*.json) into the PHASE4_RESULTS blocks of
flybrain/README.md, PLAN.md and plan.html. Usage: python render_phase4_docs.py "<interpretation>" """
import glob, json, re, sys
interp = sys.argv[1] if len(sys.argv) > 1 else ""
M = {}
for f in sorted(glob.glob("data/match_*.json")):
    tag = f.split("match_")[1][:-5]
    if tag.startswith("smoke"): continue
    M[tag] = json.load(open(f))
order = ["random", "stockfish_depth1", "stockfish_depth2", "stockfish1320"]
rows = [(k, M[k]) for k in order if k in M] + [(k, v) for k, v in M.items() if k not in order]
pct = lambda x: f"{100 * x:.0f}%"
def elo_txt(m):
    lo, hi = m["elo_diff_95ci"]; d = m["elo_diff"]
    return f"{d:+.0f} ({lo:+.0f} to {hi:+.0f})"
md = ["| opponent | games | W / D / L | score (95% CI) | Elo difference (95% CI) | fly's rating if anchored | mean plies | how games ended |", "|---|---|---|---|---|---|---|---|"]
html = ['<div class="tablewrap"><table><tr><th>Opponent</th><th>Games</th><th>W / D / L</th><th>Score (95% CI)</th><th>Elo difference (95% CI)</th><th>Rating if anchored</th><th>Mean plies</th><th>How games ended</th></tr>']
for k, m in rows:
    ends = ", ".join(f"{n} {t}" for t, n in sorted(m["terminations"].items(), key=lambda x: -x[1]))
    anch = f"{m['fly_elo_estimate']:.0f} ({m['fly_elo_95ci'][0]:.0f} to {m['fly_elo_95ci'][1]:.0f})" if m.get("fly_elo_estimate") is not None else "—"
    md.append(f"| {m['opponent']} | {m['games']} | {m['wins']} / {m['draws']} / {m['losses']} | {pct(m['score'])} ({pct(m['score_95ci'][0])} to {pct(m['score_95ci'][1])}) | {elo_txt(m)} | {anch} | {m['mean_plies']:.0f} | {ends} |")
    html.append(f'<tr><td>{m["opponent"]}</td><td class="m">{m["games"]}</td><td class="m">{m["wins"]} / {m["draws"]} / {m["losses"]}</td><td class="m">{pct(m["score"])} ({pct(m["score_95ci"][0])} to {pct(m["score_95ci"][1])})</td><td class="m">{elo_txt(m)}</td><td class="m">{anch}</td><td class="m">{m["mean_plies"]:.0f}</td><td>{ends}</td></tr>')
html.append("</table></div>")
readme = f"""**Results** (argmax policy with the anti-repetition rule, four random opening plies, colours
alternating; Elo difference from the match score, interval from the per-game outcomes):

{chr(10).join(md)}

{interp}
"""
plan_md = "Results: " + "; ".join(f"vs {m['opponent']}: {m['wins']}/{m['draws']}/{m['losses']} ({pct(m['score'])}, Elo {elo_txt(m)})" for k, m in rows) + ".\n" + interp
plan_html = f"""<p><b>Results, 17 Sep 2026.</b> {interp}</p>
{chr(10).join(html)}"""
def fill(path, block):
    t = open(path).read()
    t = re.sub(r"<!-- PHASE4_RESULTS -->.*?(?=\n)", "", t, count=0) if "<!-- PHASE4_RESULTS:begin -->" in t else t
    if "<!-- PHASE4_RESULTS:begin -->" in t:
        t = re.sub(r"<!-- PHASE4_RESULTS:begin -->.*?<!-- PHASE4_RESULTS:end -->", lambda _: "<!-- PHASE4_RESULTS:begin -->\n" + block + "\n<!-- PHASE4_RESULTS:end -->", t, flags=re.S)
    else:
        t = t.replace("<!-- PHASE4_RESULTS -->", "<!-- PHASE4_RESULTS:begin -->\n" + block + "\n<!-- PHASE4_RESULTS:end -->")
    open(path, "w").write(t)
fill("flybrain/README.md", readme); fill("PLAN.md", plan_md); fill("plan.html", plan_html)
t = open("plan.html").read()
t = t.replace('<p><b>Status.</b> The fly is a UCI engine now (one forward pass per move, legal-move mask, promotion head, anti-repetition rule) and 200-game matches against a random mover and Stockfish at its 1320 floor are running on the Mac with the recovered step-11,600 weights.</p>', '')
open("plan.html", "w").write(t)
print("rendered", [k for k, _ in rows])
