"""Render the Phase 2 probe results (data/probe.json) into the RESULTS blocks of
flybrain/README.md, PLAN.md and plan.html, and set the phase status lines."""
import json, re
r = json.load(open("data/probe.json")); S = r["stages"]
rows = [("photoreceptor_input", "Photoreceptor input (the stimulus itself)"), ("lamina_R", "Lamina L1–L5, right"), ("medulla_lobula_R", "Medulla + lobula, right (all optic-lobe intrinsic)"),
        ("visual_projection_R", "Visual projection neurons, right"), ("central_brain", "Central brain intrinsic"), ("descending", "Descending neurons")]
pct = lambda x: (f"{100 * x:.2f}%" if 0.999 < x < 0.99995 else f"{100 * x:.1f}%")
def cell(v, key):
    return pct(v[key]["square_acc"]) if key in v else "—"
def ret(v, key):
    return pct(v["retinotopic"][key]["square_acc"]) if "retinotopic" in v else "—"
md = ["| stage | neurons | generic readout, random placements | generic, random play | retinotopic readout, random placements | retinotopic, random play | side to move |", "|---|---|---|---|---|---|---|"]
html = ['<div class="tablewrap"><table><tr><th>Stage</th><th>Neurons</th><th>Generic readout, random placements</th><th>Generic, random play</th><th>Retinotopic readout, random placements</th><th>Retinotopic, random play</th><th>Side to move</th></tr>']
for k, name in rows:
    v = S[k]
    md.append(f"| {name} | {v['features']:,} | **{cell(v, 'test_random_placements')}** | {cell(v, 'test_random_play')} | **{ret(v, 'test_random_placements')}** | {ret(v, 'test_random_play')} | {pct(v['test_random_placements']['turn_acc'])} |")
    html.append(f'<tr><td>{name}</td><td class="m">{v["features"]:,}</td><td class="m"><b>{cell(v, "test_random_placements")}</b></td><td class="m">{cell(v, "test_random_play")}</td><td class="m"><b>{ret(v, "test_random_placements")}</b></td><td class="m">{ret(v, "test_random_play")}</td><td class="m">{pct(v["test_random_placements"]["turn_acc"])}</td></tr>')
html.append("</table></div>")
m = S["medulla_lobula_R"]; g_r, g_p = m["test_random_placements"]["square_acc"], m["test_random_play"]["square_acc"]
t_r, t_p = m["retinotopic"]["test_random_placements"]["square_acc"], m["retinotopic"]["test_random_play"]["square_acc"]
d_r, d_p = S["descending"]["test_random_placements"]["square_acc"], S["descending"]["test_random_play"]["square_acc"]
i_r = S["photoreceptor_input"]["test_random_placements"]["square_acc"]
met = t_r >= 0.995
verdict_md = ("**Verdict: target met.**" if met else "**Verdict: target not met yet.**")
narrative = (f"The plan asks for more than 99.5% of squares read from the medulla and lobula by a linear readout. "
             f"The generic readout, principal components of the whole stage, gives {pct(g_r)} on random placements and {pct(g_p)} on realistic positions "
             f"(kept {m['components']} components, {m['scaling']} scaling, penalty {m['weight_decay']:g}); the retinotopic readout, each square read from the neurons of its own eye columns, gives {pct(t_r)} / {pct(t_p)}. "
             f"Of the two levers, the training set was the one that mattered: at 3,072 boards the generic readout reached 91.6%, at {r['n_train']:,} boards {pct(g_r)}, with the probe's component floor, scaling and penalty chosen on the validation split. "
             f"The receptor-rate cap made no difference (0.8, 0.5 and 0.3 gave the same retinotopic accuracy on a pilot) and stays at 0.8. "
             f"The stimulus itself reads at {pct(i_r)}, the lamina at {pct(S['lamina_R']['test_random_placements']['square_acc'])} and the visual projection neurons at {pct(S['visual_projection_R']['test_random_placements']['square_acc'])}: the whole board leaves the optic lobe intact. "
             f"It then thins out in the untrained central brain ({pct(S['central_brain']['test_random_placements']['square_acc'])}) and reaches the descending neurons at {pct(d_r)} ({pct(d_p)} realistic), the baseline Phase 3 must beat.")
readme = f"""**Results** ({r['n_boards']:,} positions, {r['n_train']:,} for training, 24 ticks, gain 1.5; a readout that
always answers "empty" scores 40% on random placements and about 44% on random play):

{chr(10).join(md)}

{verdict_md} {narrative}"""
plan_md = f"""Result: {'target met' if met else 'target not met yet'}. Retinotopic linear readout of the untrained medulla + lobula: {pct(t_r)} of
squares on random placements ({pct(t_p)} realistic); generic principal-component readout {pct(g_r)} / {pct(g_p)}
(the stimulus itself reads at {pct(i_r)}). Descending neurons {pct(d_r)} / {pct(d_p)} before
any training. {r['n_train']:,} training boards; receptor-rate cap left at 0.8 (0.5 and 0.3 made no difference).
Table and details in flybrain/README.md."""
plan_html = f"""<p><b>Result, 15 Sep 2026: {'target met' if met else 'target not met yet'}.</b> {narrative}</p>
{chr(10).join(html)}"""
def fill(path, block):
    t = open(path).read()
    t = re.sub(r"<!-- RESULTS:begin -->.*?<!-- RESULTS:end -->", lambda _: "<!-- RESULTS:begin -->\n" + block + "\n<!-- RESULTS:end -->", t, flags=re.S)
    open(path, "w").write(t)
fill("flybrain/README.md", readme); fill("PLAN.md", plan_md); fill("plan.html", plan_html)
status = "DONE 2026-09-15" if met else "BUILT 2026-09-15, target not met yet"
t = open("PLAN.md").read(); t = re.sub(r"### 2 — Show it the board \(week 3\) — .*\(see flybrain/README.md\)", f"### 2 — Show it the board (week 3) — {status} (see flybrain/README.md)", t); open("PLAN.md", "w").write(t)
t = open("plan.html").read()
t = re.sub(r'<span class="when[^"]*">[^<]*</span></div>\n      <p class="goal">The board reaches', f'<span class="when{" done" if met else ""}">{"done · 15 Sep 2026" if met else "built · target not met yet"}</span></div>\n      <p class="goal">The board reaches', t)
t = re.sub(r'<span>Show it the board<span class="w">[^<]*</span>', f'<span>Show it the board<span class="w">{"done" if met else pct(t_r) + " of 99.5%"}</span>', t)
open("plan.html", "w").write(t)
t = open("README.md").read(); t = re.sub(r"\| 2 \| Show it the board \|[^|]*\|", f"| 2 | Show it the board | {'done: the medulla reads ' + pct(t_r) + ' of squares' if met else 'built; medulla reads ' + pct(t_r) + ' of squares, target 99.5%'} |", t)
t = t.replace("| 3 | Teach it the rules | |", "| 3 | Teach it the rules | next |") if met else t
open("README.md", "w").write(t)
print("rendered:", "target met" if met else "target not met", f"retinotopic {pct(t_r)}/{pct(t_p)} generic {pct(g_r)}/{pct(g_p)}")
