"""Fill the PHASE7_RESULTS blocks of flybrain/README.md, PLAN.md and plan.html from the
baseline-versus-RL match files. Usage: python render_phase7_docs.py
"""
import glob, json, math, os, re

ARMS = [("greedy", "greedy capture bot"), ("random", "random mover"), ("sfd1", "Stockfish 17, depth 1")]
rows = []
for tag, name in ARMS:
    f_b, f_r = f"data/match_{tag}_baseline.json", f"data/match_{tag}_rl.json"
    if not (os.path.exists(f_b) and os.path.exists(f_r)):
        continue
    b, r = json.load(open(f_b)), json.load(open(f_r))
    seb = (b["score_95ci"][1] - b["score"]) / 1.96
    ser = (r["score_95ci"][1] - r["score"]) / 1.96
    d = r["score"] - b["score"]; se = math.sqrt(seb ** 2 + ser ** 2)
    rows.append({"name": name, "b": b, "r": r, "d": d, "se": se, "sigma": d / se if se else 0.0})

def verdict(x):
    return "no change" if abs(x["sigma"]) < 2 else ("better" if x["sigma"] > 0 else "worse")

md = ["| opponent | weights | games | W | D | L | score | change |", "|---|---|---|---|---|---|---|---|"]
html = ['<div class="tablewrap"><table><tr><th>Opponent</th><th>Weights</th><th>Games</th><th>W</th><th>D</th>'
        '<th>L</th><th>Score</th><th>Change</th></tr>']
for x in rows:
    for arm, m in (("supervised", x["b"]), ("after RL", x["r"])):
        chg = "—" if arm == "supervised" else f"{x['d']:+.3f} ± {x['se']:.3f} ({x['sigma']:+.1f} SE, {verdict(x)})"
        md.append(f"| {x['name'] if arm == 'supervised' else ''} | {arm} | {m['games']} | {m['wins']} | {m['draws']} "
                  f"| {m['losses']} | {m['score']:.3f} | {chg} |")
        html.append(f'<tr><td>{x["name"] if arm == "supervised" else ""}</td><td>{arm}</td>'
                    f'<td class="m">{m["games"]}</td><td class="m">{m["wins"]}</td><td class="m">{m["draws"]}</td>'
                    f'<td class="m">{m["losses"]}</td><td class="m">{m["score"]:.3f}</td><td class="m">{chg}</td></tr>')
html.append("</table></div>")
block_md = "\n".join(md)
block_html = "\n".join(html)

def fill(path, block):
    t = open(path).read()
    if "<!-- PHASE7_RESULTS:begin -->" in t:
        t = re.sub(r"<!-- PHASE7_RESULTS:begin -->.*?<!-- PHASE7_RESULTS:end -->",
                   lambda _: "<!-- PHASE7_RESULTS:begin -->\n" + block + "\n<!-- PHASE7_RESULTS:end -->", t, flags=re.S)
    elif "<!-- PHASE7_RESULTS -->" in t:
        t = t.replace("<!-- PHASE7_RESULTS -->", "<!-- PHASE7_RESULTS:begin -->\n" + block + "\n<!-- PHASE7_RESULTS:end -->")
    else:
        return False
    open(path, "w").write(t)
    return True

for path, block in (("flybrain/README.md", block_md), ("PLAN.md", block_md), ("plan.html", block_html)):
    print(("filled " if fill(path, block) else "no marker in ") + path)
print(f"{len(rows)} matchups")
