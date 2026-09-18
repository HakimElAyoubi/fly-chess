"""Draw the ablation map: how much the fly's move choice degrades when each part of its brain is
silenced. Writes data/ablation_map.svg and prints the markdown table.
Usage: python render_ablation.py
"""
import json
import math

d = json.load(open("data/ablation.json"))
base = d["intact"]["top1"]; n = d["n_positions"]
se = math.sqrt(base * (1 - base) / n)                       # one standard error on the intact score
rows = sorted(d["groups"].items(), key=lambda kv: -kv[1]["top1_drop"])

# a group is "load bearing" if silencing it costs more than three standard errors
THRESH = 3 * se
FAMILY = {
    "photoreceptors": "eye", "lamina (L1-L5)": "eye",
    "optic lobe (intrinsic)": "visual", "distal medulla (Dm)": "visual", "medulla intrinsic (Mi)": "visual",
    "transmedullary (Tm, TmY)": "visual", "lobula columnar (LC, LPLC)": "visual",
    "motion detectors (T4, T5)": "visual", "medulla tangential (Pm, Li)": "visual",
    "visual projection neurons": "visual", "visual centrifugal": "visual",
    "central brain (intrinsic)": "central", "descending neurons": "output", "ascending neurons": "central",
    "brain sensory axons": "other sense", "mechanosensory": "other sense", "gustatory": "other sense",
    "olfactory receptor neurons": "other sense", "antennal lobe projection": "other sense",
    "antennal lobe local": "other sense",
    "mushroom body: Kenyon cells": "higher brain", "mushroom body: outputs": "higher brain",
    "mushroom body: dopaminergic": "higher brain", "lateral horn": "higher brain",
    "central complex": "higher brain", "CX: ring neurons (ER)": "higher brain",
    "CX: compass (EPG, PEG, PEN)": "higher brain", "CX: fan-shaped body": "higher brain",
}
COLOUR = {"eye": "#1A7F9C", "visual": "#4FA3BE", "output": "#B4562E", "central": "#7BA05B",
          "other sense": "#9B8FB4", "higher brain": "#C9A227"}

print("| silenced | neurons | top-1 after | change |")
print("|---|---|---|---|")
for k, v in rows:
    sig = "" if v["top1_drop"] > THRESH else "  (within noise)"
    dp = 100 * v["top1_drop"]
    print(f"| {k} | {v['neurons']:,} | {100*v['top1']:.1f}% | {'−' if dp >= 0 else '+'}{abs(dp):.1f} pts{sig} |")

W, ROW, PAD_L, PAD_T = 760, 21, 250, 54
H = PAD_T + ROW * len(rows) + 62
scale = (W - PAD_L - 130) / max(v["top1_drop"] for _, v in rows)
out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
       f'aria-label="How far the fly\'s agreement with Stockfish falls when each part of its brain is silenced. '
       f'The visual pathway and the descending neurons are load bearing; the mushroom body, central complex and '
       f'lateral horn cost nothing.">']
out.append(f'<text x="14" y="24" font-size="14" font-weight="600" fill="currentColor">Silencing each part of the brain</text>')
out.append(f'<text x="14" y="42" font-size="11.5" fill="currentColor" opacity="0.7">'
           f'drop in agreement with Stockfish\'s move, from an intact {100*base:.1f}% · {n:,} held-out positions</text>')
for k, v in rows:
    i = rows.index((k, v)); y = PAD_T + i * ROW
    fam = FAMILY.get(k, "central"); c = COLOUR[fam]
    w = v["top1_drop"] * scale
    faint = v["top1_drop"] <= THRESH
    out.append(f'<text x="{PAD_L - 8}" y="{y + 11}" font-size="11" text-anchor="end" fill="currentColor"'
               + (' opacity="0.55"' if faint else '') + f'>{k}</text>')
    out.append(f'<rect x="{PAD_L}" y="{y + 2}" width="{max(w, 0.6):.1f}" height="{ROW - 7}" rx="2" fill="{c}"'
               + (' opacity="0.3"' if faint else '') + '/>')
    dp = 100 * v["top1_drop"]
    label = ("−" if dp >= 0 else "+") + f"{abs(dp):.1f}" + (" (noise)" if faint else "")
    out.append(f'<text x="{PAD_L + max(w, 0.6) + 6:.1f}" y="{y + 11}" font-size="10.5" fill="currentColor" opacity="0.75">{label}</text>')
out.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + ROW * len(rows)}" stroke="currentColor" stroke-opacity="0.4"/>')
lx = PAD_L; ly = H - 26
for fam, c in COLOUR.items():
    out.append(f'<rect x="{lx}" y="{ly - 8}" width="9" height="9" rx="2" fill="{c}"/>')
    out.append(f'<text x="{lx + 13}" y="{ly}" font-size="10.5" fill="currentColor">{fam}</text>')
    lx += 22 + 6.3 * len(fam)
out.append("</svg>")
open("data/ablation_map.svg", "w").write("\n".join(out))
print(f"\nwrote data/ablation_map.svg ({len(rows)} groups, noise threshold {100*THRESH:.2f} points)")
