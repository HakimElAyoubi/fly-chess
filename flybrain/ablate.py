"""Phase 5, part 1: where in the brain does the chess happen?

Silence one neuropil at a time and re-measure how well the fly picks Stockfish's move. A region
whose removal costs nothing was not being used; a region whose removal is catastrophic is load
bearing. The result is a map over the whole brain, which is also what the demo's brain inset
should highlight.

A neuron belongs to a neuropil if most of its synapses are there, taken from the dataset's own
per-region synapse counts. Silencing means clamping those neurons' firing rate to zero at every
tick, so they neither respond nor pass anything on.

    python -m flybrain.ablate --weights data/train_gpu/model_step11600.pt --n 2048

Writes data/ablation.json.
"""
import argparse
import gzip
import json
import time
import numpy as np
import pandas as pd
import pyarrow.feather as pf
import torch
import chess
from .graph import load, DATA
from .eye import Eye
from .policy import FlyPolicy, variant_of, load_into, legal_mask, move_index
from .progress import write as progress

BODY_STATS = DATA / "body-stats-male-cns-v1.0-minconf-0.5.feather"


def neuropil_of_neuron(meta, min_synapses=50):
    """Assign each neuron to the neuropil holding most of its synapses.
    Returns a Series of region names indexed by the model's neuron index."""
    cols = pf.read_table(BODY_STATS).schema.names
    roi_cols = [c for c in cols if c.endswith("_pre") or c.endswith("_post")]
    keep = ["body"] + roi_cols if "body" in cols else roi_cols
    t = pf.read_table(BODY_STATS, columns=keep).to_pandas()
    body_col = "body" if "body" in t.columns else t.columns[0]
    regions = sorted({c.rsplit("_", 1)[0] for c in roi_cols})
    tot = pd.DataFrame({r: t.get(f"{r}_pre", 0) + t.get(f"{r}_post", 0) for r in regions}, index=t[body_col])
    tot = tot.loc[tot.index.intersection(meta.bodyId.values)]
    best = tot.idxmax(axis=1); total = tot.max(axis=1)
    best[total < min_synapses] = "unassigned"
    s = pd.Series("unassigned", index=meta.index)
    m = meta.bodyId.map(best)
    s[m.notna().to_numpy()] = m.dropna().to_numpy()
    return s


@torch.no_grad()
def score(model, eye, recs, batch, silence=None, device="cpu"):
    """Top-1 agreement with Stockfish's move, and the legal rate, with `silence` clamped off."""
    mask = None
    if silence is not None and silence.any():
        mask = torch.ones(model.N, 1, device=device)
        mask[torch.as_tensor(np.flatnonzero(silence), device=device)] = 0.0
    top1 = legal = n = 0
    for i in range(0, len(recs), batch):
        chunk = recs[i:i + batch]
        boards = [chess.Board(r["fen"]) for r in chunk]
        I = torch.from_numpy(np.stack([eye.encode(b) for b in boards], 1)).to(device)
        logits, _, _, _ = model(I, silence=mask)
        y = torch.tensor([move_index(chess.Move.from_uci(r["sf_move"]))[0] for r in chunk], device=device)
        lm = legal_mask(boards, device)
        legal += int((lm[torch.arange(len(boards), device=device), logits.argmax(1)] == 0).sum())
        top1 += int(((logits + lm).argmax(1) == y).sum())
        n += len(chunk)
    return top1 / n, legal / n


def run(weights, positions, n_positions, device, batch, min_neurons=50):
    edge, ck = variant_of(weights, device)
    model = FlyPolicy(device=device, ticks=24, edge_gains=edge)
    load_into(model, ck["model"], weights); model.eval()
    W, meta = load(); eye = Eye(W, meta)
    region = neuropil_of_neuron(meta)
    counts = region.value_counts()
    targets = [r for r in counts.index if r != "unassigned" and counts[r] >= min_neurons]
    print(f"{len(targets)} neuropils with at least {min_neurons} neurons; "
          f"{int(counts.get('unassigned', 0))} neurons unassigned", flush=True)
    recs = [json.loads(l) for _, l in zip(range(n_positions), gzip.open(positions, "rt"))]

    base_top1, base_legal = score(model, eye, recs, batch, None, device)
    print(f"intact: top1 {base_top1:.4f}  legal {base_legal:.4f}", flush=True)
    out = {"weights": weights, "n_positions": len(recs), "intact": {"top1": base_top1, "legal": base_legal}, "regions": {}}
    t0 = time.time()
    for i, r in enumerate(targets, 1):
        sil = (region == r).to_numpy()
        t1, lg = score(model, eye, recs, batch, sil, device)
        out["regions"][r] = {"neurons": int(sil.sum()), "top1": t1, "legal": lg,
                             "top1_drop": base_top1 - t1, "legal_drop": base_legal - lg}
        progress(i, len(targets), t0, "ablation")
        print(f"  {r:22s} {int(sil.sum()):6d} neurons silenced -> top1 {t1:.4f} ({t1-base_top1:+.4f})  legal {lg:.4f} ({lg-base_legal:+.4f})", flush=True)
        json.dump(out, open(DATA / "ablation.json", "w"), indent=1)
    worst = sorted(out["regions"].items(), key=lambda kv: -kv[1]["top1_drop"])[:8]
    print("\nmost load-bearing regions:")
    for k, v in worst:
        print(f"  {k:22s} top1 falls {100*v['top1_drop']:5.2f} points  ({v['neurons']} neurons)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DATA / "train_gpu" / "model_step11600.pt"))
    ap.add_argument("--positions", default=str(DATA / "positions_all_sf.jsonl.gz"))
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-neurons", type=int, default=50)
    a = ap.parse_args()
    run(a.weights, a.positions, a.n, a.device, a.batch, a.min_neurons)
