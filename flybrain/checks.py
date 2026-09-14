"""Biology checks for the untrained digital fly, and calibration of its gain.

Model: graded units (tanh), inputs of every neuron normalised to sum to 1 in magnitude, one
global gain. Activity is measured relative to rest; the resting state is stable for gain <= 1
and empirically stays sparse up to about 1.5.

Check A (taste -> feeding), after Shiu et al. 2024: drive the labellar sugar taste neurons.
  - the named second-order sugar neurons of Shiu et al. 2022 respond strongly;
  - the proboscis motor neuron MN9 is pushed above rest;
  - MN9's response to sugar beats its response to every one of 20 random sensory sets of the
    same size (other labellar taste neurons, brain mechanosensory neurons), z >= 3;
  - MN9 ranks in the top 3% of all responding neurons; fewer than 10% of neurons respond.

Check B (light -> motion vision): drive the R1-R6 photoreceptors of the right eye.
  - lamina L1/L2, motion detectors T4/T5 and lobula plate tangential cells of the RIGHT optic
    lobe respond at least 5x (3x for the tangential cells) more than those of the left;
  - fewer than 10% of neurons respond.
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
from .graph import load, DATA
from .model import FlyBrain

SUGAR_2ND_ORDER = ["G2N-1", "Phantom", "Usnea", "Rattle", "Zorro"]      # Shiu et al. 2022 sugar interneurons
PROBOSCIS_MNS = ["MN9", "MN1", "MN4a", "MN4b", "MN6", "MN7", "MN8", "MN10", "MN11D", "MN11V", "MN12D", "MN13"]
GAINS = (0.5, 0.9, 1.5, 2.5)
N_CONTROLS = 20


def pick_sets(W, meta, rng):
    t = meta.type.fillna("")
    syn = meta.synonyms.fillna("")
    sugar2 = meta.index[syn.str.contains("Shiu 2022") & syn.str.contains("|".join(SUGAR_2ND_ORDER), regex=True)].to_numpy()
    labellar = meta.index[(meta["class"] == "gustatory") & meta.subclass.isin(["labellar bristle", "taste peg", "pharyngeal sensillum"])].to_numpy()
    # sugar GRNs = labellar taste neurons with direct synapses onto the sugar second-order neurons
    sub = W[sugar2][:, labellar]                       # post = sugar2, pre = labellar
    syn_to_sugar2 = np.asarray(np.abs(sub).sum(axis=0)).ravel()
    sugar_grn = labellar[syn_to_sugar2 >= 2]
    other_grn = labellar[syn_to_sugar2 == 0]
    mech = meta.index[(meta.superclass == "cb_sensory") & (meta["class"].fillna("").str.startswith("mechanosensory"))].to_numpy()
    sets = {
        "sugar_grn": sugar_grn, "other_labellar_grn_pool": other_grn, "mechanosensory_pool": mech,
        "sugar_2nd_order": sugar2,
        "MN9": meta.index[t == "MN9"].to_numpy(),
        "proboscis_mns": meta.index[t.isin(PROBOSCIS_MNS)].to_numpy(),
    }
    side = meta.side.fillna("")
    r16 = (t == "R1-R6")
    sets["R1-6_right"] = meta.index[r16 & (side == "R")].to_numpy()
    sets["R1-6_left"] = meta.index[r16 & (side == "L")].to_numpy()
    for name, pat in [("T4T5", r"^(T4|T5)[a-d]$"), ("LPTC", r"^(HS[NES]|VS|VS[m]?|VST[12]|H2|CH)$"), ("L1L2", r"^(L1|L2)$")]:
        m = t.str.match(pat)
        sets[name + "_right"] = meta.index[m & (side == "R")].to_numpy()
        sets[name + "_left"] = meta.index[m & (side == "L")].to_numpy()
    return sets


def summarize(frames, sets, names, stim):
    """Responses at the last tick, relative to the peak response outside the stimulated set."""
    r = frames[-1][:, 0].numpy()
    mask = np.ones(len(r), bool); mask[stim] = False
    peak = float(np.abs(r[mask]).max()) or 1.0
    out = {n: float(np.abs(r[sets[n]]).mean() / peak) if len(sets[n]) else float("nan") for n in names}
    out["MN9_signed"] = float(r[sets["MN9"]].mean() / peak)
    ranks = np.argsort(-np.abs(np.where(mask, r, 0.0)))
    out["MN9_rank"] = int(np.where(np.isin(ranks, sets["MN9"]))[0].min()) + 1
    out["frac_active"] = float((np.abs(r[mask]) > 0.05 * peak).mean())
    out["peak_abs"] = peak
    return out


def run_checks(gains=GAINS, T=48, device="cpu", verbose=True, n_controls=N_CONTROLS):
    W, meta = load()
    rng = np.random.default_rng(0)
    sets = pick_sets(W, meta, rng)
    N = len(meta)
    if verbose:
        print("sugar GRNs:", len(sets["sugar_grn"]), "types:", meta.loc[sets["sugar_grn"]].type.value_counts().head(6).to_dict())
        print("sugar 2nd-order neurons:", sorted(set(meta.loc[sets["sugar_2nd_order"]].type)))
        print("R1-6 right/left:", len(sets["R1-6_right"]), len(sets["R1-6_left"]),
              "| T4T5 R/L:", len(sets["T4T5_right"]), len(sets["T4T5_left"]),
              "| LPTC R/L:", len(sets["LPTC_right"]), len(sets["LPTC_left"]))
    results = []
    for gain in gains:
        cfg = dict(alpha=gain, nonlin="tanh", normalize=True)
        brain = FlyBrain(W, device=device, **cfg)
        row = {"config": cfg}
        _, fr = brain.run(brain.current(sets["sugar_grn"]), T=T)
        a = row["sugar_grn"] = summarize(fr, sets, ["MN9", "proboscis_mns", "sugar_2nd_order"], sets["sugar_grn"])
        _, fr = brain.run(brain.current(sets["R1-6_right"]), T=T)
        v = row["light_right_eye"] = summarize(fr, sets, ["T4T5_right", "T4T5_left", "LPTC_right", "LPTC_left", "L1L2_right", "L1L2_left"], sets["R1-6_right"])
        row["pass_B"] = bool(v["T4T5_right"] > 5 * v["T4T5_left"] and v["LPTC_right"] > 3 * v["LPTC_left"] and v["L1L2_right"] > 5 * v["L1L2_left"]
                             and v["T4T5_right"] > 0.01 and v["frac_active"] < 0.10)
        row["pre_A"] = bool(a["MN9_signed"] > 0 and a["sugar_2nd_order"] >= 0.3 and a["MN9_rank"] <= 0.03 * N and a["frac_active"] < 0.10)
        results.append(row)
        if verbose:
            print(f"\ngain={gain:g}")
            print(f"  A  sugar: MN9={a['MN9_signed']:+.4f} (rank {a['MN9_rank']}) probMNs={a['proboscis_mns']:.4f} 2nd-order={a['sugar_2nd_order']:.3f} active={a['frac_active']:.2%}  -> {'candidate' if row['pre_A'] else 'fail'}")
            print(f"  B  light R eye: T4T5 R={v['T4T5_right']:.3f} L={v['T4T5_left']:.4f} | LPTC R={v['LPTC_right']:.4f} L={v['LPTC_left']:.4f}"
                  f" | L1L2 R={v['L1L2_right']:.3f} L={v['L1L2_left']:.4f} | active={v['frac_active']:.2%}  -> {'PASS' if row['pass_B'] else 'fail'}")
    cands = [r for r in results if r["pass_B"] and r["pre_A"]]
    if not cands:
        return results, None, sets
    # among candidates take the strongest motor response, then run the control distribution
    best = max(cands, key=lambda r: r["sugar_grn"]["MN9_signed"])
    brain = FlyBrain(W, device=device, **best["config"])
    n = len(sets["sugar_grn"])
    ctrl = []
    for i in range(n_controls):
        pool = sets["other_labellar_grn_pool"] if i % 2 == 0 else sets["mechanosensory_pool"]
        s = rng.choice(pool, size=min(n, len(pool)), replace=False)
        _, fr = brain.run(brain.current(s), T=T)
        ctrl.append(summarize(fr, sets, ["MN9"], s)["MN9_signed"])
    ctrl = np.array(ctrl)
    a = best["sugar_grn"]["MN9_signed"]
    z = (a - ctrl.mean()) / (ctrl.std() + 1e-12)
    best["controls"] = {"n": int(n_controls), "MN9_signed": ctrl.tolist(), "mean": float(ctrl.mean()), "std": float(ctrl.std()), "max": float(ctrl.max()), "z": float(z)}
    best["pass_A"] = bool(a > ctrl.max() and z >= 3)
    if verbose:
        print(f"\ncontrols at gain {best['config']['alpha']:g}: MN9 response mean {ctrl.mean():+.4f} sd {ctrl.std():.4f} max {ctrl.max():+.4f}; sugar {a:+.4f}; z={z:.1f}"
              f"  -> check A {'PASS' if best['pass_A'] else 'fail'}")
    chosen = best["config"] if best["pass_A"] else None
    return results, chosen, sets


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    results, chosen, sets = run_checks(device=device)
    print("\nchosen config:", chosen)
    json.dump({"results": results, "chosen_config": chosen, "dt_over_tau": 0.25,
               "set_sizes": {k: int(len(v)) for k, v in sets.items()}},
              open(DATA / "checks.json", "w"), indent=1)
