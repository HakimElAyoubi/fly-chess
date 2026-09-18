"""Phase 5, part 3: what did training actually change about the wiring?

The connectome fixes which neurons exist, who talks to whom, and each neuron's sign. Training
was allowed to change only how loudly each connection speaks. This asks where those changes
landed: whether they cluster in particular cell types or pathways, whether strong anatomical
connections were treated differently from weak ones, and whether the learned network still
resembles the measured one.

    python -m flybrain.gains [--weights data/train_gpu/model_step11600.pt]

Writes data/gains.json and prints the summary.
"""
import argparse
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from .graph import load, DATA


def edge_table(W, meta):
    """One row per connection, in the same order as the model's gain vector."""
    W = W.tocsr()
    post = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
    pre = W.indices
    syn = np.abs(W.data)
    return pre, post, syn


def summarise(by, gain, syn, min_edges=2000, top=12):
    """Median gain per group, weighted by nothing: we want the typical connection, not the loudest."""
    df = pd.DataFrame({"k": by, "g": gain, "s": syn})
    agg = df.groupby("k").agg(edges=("g", "size"), median_gain=("g", "median"),
                              frac_up=("g", lambda x: float((x > 1.1).mean())),
                              frac_down=("g", lambda x: float((x < 0.9).mean())),
                              synapses=("s", "sum"))
    agg = agg[agg.edges >= min_edges].sort_values("median_gain")
    return agg


def run(weights):
    W, meta = load()
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    g = ck["model"]["log_edge_gain"].exp().numpy()
    pre, post, syn = edge_table(W, meta)
    assert len(g) == len(pre), f"{len(g)} gains but {len(pre)} edges"
    out = {"weights": weights, "step": ck.get("step"), "edges": int(len(g))}

    q = np.percentile(g, [1, 5, 25, 50, 75, 95, 99])
    out["gain_quantiles"] = {f"p{p}": float(v) for p, v in zip([1, 5, 25, 50, 75, 95, 99], q)}
    out["frac_changed_10pct"] = float((np.abs(g - 1) > 0.1).mean())
    out["frac_silenced"] = float((g < 0.5).mean())
    out["frac_amplified_2x"] = float((g > 2).mean())
    print(f"gains: median {np.median(g):.3f}, 5th-95th {q[1]:.3f}-{q[5]:.3f}, "
          f"{100*out['frac_changed_10pct']:.1f}% changed >10%, {100*out['frac_silenced']:.1f}% halved, "
          f"{100*out['frac_amplified_2x']:.1f}% doubled")

    # did training treat strong anatomical connections differently from weak ones?
    bins = [1, 2, 3, 5, 10, 20, 50, 10**9]
    lab = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50+"]
    idx = np.digitize(syn, bins[1:-1], right=False)
    print("\nby how many synapses the connection has in the fly:")
    out["by_synapse_count"] = {}
    for i, name in enumerate(lab):
        m = idx == i
        if m.sum() < 100: continue
        out["by_synapse_count"][name] = {"edges": int(m.sum()), "median_gain": float(np.median(g[m])),
                                         "frac_up": float((g[m] > 1.1).mean()), "frac_down": float((g[m] < 0.9).mean())}
        print(f"  {name:>6s} synapses: {m.sum():9,d} edges  median gain {np.median(g[m]):.3f}  "
              f"turned up {100*(g[m] > 1.1).mean():4.1f}%  turned down {100*(g[m] < 0.9).mean():4.1f}%")

    # where the changes land, by the class of neuron receiving the connection
    sc = meta.superclass.fillna("unknown").to_numpy()
    for side, name in ((post, "receiving neuron"), (pre, "sending neuron")):
        agg = summarise(sc[side], g, syn)
        out[f"by_{name.split()[0]}_superclass"] = json.loads(agg.to_json(orient="index"))
        print(f"\nby the {name}'s class (median gain, quietest first):")
        for k, r in agg.head(6).iterrows():
            print(f"  {k:24s} {int(r.edges):9,d} edges  median {r.median_gain:.3f}  up {100*r.frac_up:4.1f}%  down {100*r.frac_down:4.1f}%")
        for k, r in agg.tail(3).iterrows():
            print(f"  {k:24s} {int(r.edges):9,d} edges  median {r.median_gain:.3f}  up {100*r.frac_up:4.1f}%  down {100*r.frac_down:4.1f}%")

    # and by cell type, which is the finer-grained question
    ty = meta.type.fillna("unknown").to_numpy()
    agg = summarise(ty[post], g, syn, min_edges=5000)
    out["by_post_type"] = json.loads(agg.to_json(orient="index"))
    print("\ncell types whose inputs were turned down most / up most (>= 5,000 edges):")
    for k, r in agg.head(6).iterrows():
        print(f"  {k:20s} {int(r.edges):8,d} edges  median {r.median_gain:.3f}")
    for k, r in agg.tail(6).iterrows():
        print(f"  {k:20s} {int(r.edges):8,d} edges  median {r.median_gain:.3f}")

    # how much of the original wiring survives: correlation between measured and effective strength
    eff = syn * g
    out["spearman_syn_vs_effective"] = float(pd.Series(syn).corr(pd.Series(eff), method="spearman"))
    out["median_abs_log2_gain"] = float(np.median(np.abs(np.log2(g))))
    print(f"\nmeasured synapse count vs effective strength after training: Spearman {out['spearman_syn_vs_effective']:.3f}")
    print(f"typical connection moved by a factor of {2**out['median_abs_log2_gain']:.2f}")
    json.dump(out, open(DATA / "gains.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DATA / "train_gpu" / "model_step11600.pt"))
    a = ap.parse_args()
    run(a.weights)
