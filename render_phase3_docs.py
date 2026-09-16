"""Regenerate the Phase 3 results block in flybrain/README.md, PLAN.md and plan.html from the
GPU run log (data/run_gpu.log, stages 1 and 2) and the first run's summary
(data/train_gpu_summary.json). Usage: python render_phase3_docs.py "<one-paragraph interpretation>"
"""
import json, re, sys
interp = sys.argv[1] if len(sys.argv) > 1 else ""
rows = []
for line in open("data/run_gpu.log"):
    m = re.match(r"step\s+(\d+)\s+train loss ([\d.]+)\s+eval loss ([\d.]+)\s+legal ([\d.]+)\s+top1 ([\d.]+)\s+top3 ([\d.]+)\s+value ([\d.]+)\s+\(([\d.]+) min\)", line)
    if m and int(m.group(1)) > 3:
        rows.append([float(x) for x in m.groups()])
first = json.load(open("data/train_gpu_summary.json"))
pct = lambda x: f"{100 * x:.1f}%"
last = rows[-1]; best = {k: max(r[i] for r in rows) for k, i in (("legal", 3), ("top1", 4), ("top3", 5), ("value", 6))}
positions = lambda step: int(step) * 512 if step > 1500 else int(step) * 512   # 4 GPUs x 128
sel = [r for r in rows if int(r[0]) in (200, 800, 1500) or (int(r[0]) > 1500 and int(r[0]) % 2000 == 0)] + [rows[-1]]
sel = sorted({int(r[0]): r for r in sel}.values(), key=lambda r: r[0])
md = "\n".join(f"| {int(r[0]):,} | {positions(r[0]):,} | {r[2]:.2f} | {pct(r[3])} | {pct(r[4])} | {pct(r[5])} | {pct(r[6])} |" for r in sel)
html = "".join(f'<tr><td class="m">{int(r[0]):,}</td><td class="m">{positions(r[0]):,}</td><td class="m">{r[2]:.2f}</td><td class="m">{pct(r[3])}</td><td class="m">{pct(r[4])}</td><td class="m">{pct(r[5])}</td><td class="m">{pct(r[6])}</td></tr>' for r in sel)
svg = open("data/phase3_curve.svg").read()
f_last, f_best = first["last"], first["best"]
readme = f"""## Phase 3 results (16 Sep 2026)

**Second run, four RTX 5090s.** The per-edge variant (27.1 M parameters) trained with data
parallelism over four GPUs, 512 positions per step at 0.91 s per step, for {int(last[0]):,} steps:
{positions(last[0]):,} positions from 4.3 million distinct ones (600,000 Lichess games streamed on
the box), {last[7]:.0f} minutes of stage 2 after a 23-minute warm-up stage on 431,000 positions.
Evaluation on 512 held-out positions from games not used in training:

| step | positions seen | eval loss | legal-move rate | top-1 | top-3 | value acc |
|---|---|---|---|---|---|---|
{md}

Best values over the run: legal {pct(best['legal'])}, top-1 {pct(best['top1'])}, top-3 {pct(best['top3'])}, value {pct(best['value'])}.
Chance levels: 0.7% legal, about 3% top-1, 9% top-3, 33% value. Curves: `data/phase3_curve.svg`.

**First run, one RTX 4090** (768,000 positions, 125 minutes): legal {pct(f_last['legal'])} (best {pct(f_best['legal'])}), top-1
{pct(f_last['top1'])} (best {pct(f_best['top1'])}), top-3 {pct(f_last['top3'])}. The Mac control with per-neuron gains only: 16.8% legal, 6.2% top-1.

{interp}
"""
plan_md = f"""GPU runs: (1) one RTX 4090, 768k positions: legal {pct(f_last['legal'])}, top-1 {pct(f_last['top1'])}; (2) four RTX 5090s,
{positions(last[0]) / 1e6:.1f} M positions: legal {pct(last['legal']) if isinstance(last, dict) else pct(last[3])} (best {pct(best['legal'])}), top-1 {pct(last[4])} (best {pct(best['top1'])}),
top-3 {pct(last[5])} (best {pct(best['top3'])}), eval loss {last[2]:.2f}. Mac control (per-neuron gains): 16.8% / 6.2%.
{interp}"""
plan_html = f"""<p><b>Results, 16 Sep 2026.</b> A first run on one RTX 4090 saw 768,000 positions in two hours and reached {pct(f_last['legal'])} legal, {pct(f_last['top1'])} top-1. A second run on four RTX 5090s saw {positions(last[0]) / 1e6:.1f} million positions (4.3 million distinct) and reached <b>{pct(last[3])}</b> legal (best {pct(best['legal'])}), <b>{pct(last[4])}</b> top-1 (best {pct(best['top1'])}) and {pct(last[5])} top-3; chance is 0.7%, 3% and 9%. The Mac control that may only rescale neurons, not synapses, reaches 17% legal and 6% top-1. {interp}</p>
<div class="tablewrap"><table><tr><th>Step</th><th>Positions</th><th>Eval loss</th><th>Legal</th><th>Top-1</th><th>Top-3</th><th>Value</th></tr>{html}</table></div>
<figure>{svg}<figcaption>Learning curves of the four-GPU run on 512 held-out positions: legal-move rate, top-1 and top-3 agreement with the human move among legal moves, and value accuracy, against training steps of 512 positions.</figcaption></figure>"""
def fill(path, block):
    t = open(path).read()
    t = re.sub(r"<!-- PHASE3_RESULTS:begin -->.*?<!-- PHASE3_RESULTS:end -->", lambda _: "<!-- PHASE3_RESULTS:begin -->\n" + block + "\n<!-- PHASE3_RESULTS:end -->", t, flags=re.S)
    open(path, "w").write(t)
fill("flybrain/README.md", readme); fill("PLAN.md", plan_md); fill("plan.html", plan_html)
t = open("plan.html").read(); t = re.sub(r'<span>Teach it the rules<span class="w">[^<]*</span>', f'<span>Teach it the rules<span class="w">{pct(last[4])} top-1</span>', t); open("plan.html", "w").write(t)
t = open("README.md").read(); t = re.sub(r"\| 3 \| Teach it the rules \|[^|]*\|", f"| 3 | Teach it the rules | in progress: {pct(last[3])} legal, {pct(last[4])} top-1 after {positions(last[0]) / 1e6:.1f} M positions |", t); open("README.md", "w").write(t)
json.dump({"evals": rows, "best": best, "last": dict(zip(["step","train_loss","eval_loss","legal","top1","top3","value","minutes"], last))}, open("data/train_gpu2_summary.json", "w"), indent=1)
print(f"rendered: {len(rows)} evaluations; last step {int(last[0]):,}: legal {pct(last[3])} top-1 {pct(last[4])} top-3 {pct(last[5])} (best {pct(best['legal'])}/{pct(best['top1'])}/{pct(best['top3'])})")
