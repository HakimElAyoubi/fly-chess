"""Render Phase 3 learning curves from a training log (the eval lines of run_gpu.log or a
train log) into data/phase3_curve.svg and a markdown table on stdout.
Usage: python render_phase3.py <log> [label] [out.svg] [positions per step]"""
import re, sys
path = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else "GPU run, per-edge gains"
out_path = sys.argv[3] if len(sys.argv) > 3 else "data/phase3_curve.svg"
per_step = int(sys.argv[4]) if len(sys.argv) > 4 else 128
rows = []
for line in open(path):
    m = re.match(r"step\s+(\d+)\s+train loss ([\d.]+)\s+eval loss ([\d.]+)\s+legal ([\d.]+)\s+top1 ([\d.]+)\s+top3 ([\d.]+)\s+value ([\d.]+)\s+\(([\d.]+) min\)", line)
    if m:
        rows.append(tuple(float(x) for x in m.groups()))
if not rows:
    sys.exit("no eval lines found")
print("| step | positions seen | eval loss | legal rate | top-1 | top-3 | value acc | minutes |")
print("|---|---|---|---|---|---|---|---|")
for s, tl, el, lg, t1, t3, va, mn in rows:
    print(f"| {int(s):,} | {int(s) * per_step:,} | {el:.2f} | {100 * lg:.1f}% | {100 * t1:.1f}% | {100 * t3:.1f}% | {100 * va:.1f}% | {mn:.0f} |")
# SVG: three curves on one 0-100% axis
W, H, L, R, T, B = 720, 300, 56, 16, 20, 40
xs = [r[0] for r in rows]; xmax = max(xs)
def X(s): return L + (W - L - R) * s / xmax
def Y(v): return T + (H - T - B) * (1 - v)
series = [("legal rate", 3, "#1A7F9C"), ("top-1", 4, "#B4562E"), ("top-3", 5, "#7BA05B"), ("value acc", 6, "#8B909D")]
out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="Learning curves of the digital fly: legal-move rate, top-1 and top-3 agreement with human moves, and value accuracy against training steps">',
       '<rect width="100%" height="100%" fill="none"/>']
for v in (0.25, 0.5, 0.75, 1.0):
    out.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W - R}" y2="{Y(v):.1f}" stroke="currentColor" stroke-opacity="0.15"/>')
    out.append(f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="currentColor">{int(v * 100)}%</text>')
out.append(f'<line x1="{L}" y1="{Y(0):.1f}" x2="{W - R}" y2="{Y(0):.1f}" stroke="currentColor" stroke-opacity="0.4"/>')
tick = max(100, int(xmax // 6 // 100 * 100) or 100)
for k in range(0, int(xmax) + 1, tick):
    out.append(f'<text x="{X(k):.1f}" y="{H - B + 16}" font-size="11" text-anchor="middle" fill="currentColor">{k:,}</text>')
out.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 4}" font-size="11" text-anchor="middle" fill="currentColor" opacity="0.7">training steps (×{per_step} positions)</text>')
for name, i, colour in series:
    pts = " ".join(f"{X(r[0]):.1f},{Y(r[i]):.1f}" for r in rows)
    out.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>')
    last = rows[-1]
    out.append(f'<text x="{X(last[0]) + 4:.1f}" y="{Y(last[i]) + 4:.1f}" font-size="11" fill="{colour}">{name} {100 * last[i]:.0f}%</text>')
out.append(f'<text x="{L}" y="{T - 6}" font-size="12" fill="currentColor" font-weight="600">{label}</text></svg>')
open(out_path, "w").write("\n".join(out))
print(f"\nwrote {out_path} ({len(rows)} evaluations)")
