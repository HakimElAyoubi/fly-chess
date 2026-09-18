"""Phase 5, part 1: where in the brain does the chess happen?

Silence one functional group at a time and re-measure how well the fly picks Stockfish's move. A region
whose removal costs nothing was not being used; a region whose removal is catastrophic is load
bearing. The result is a map over the whole brain, which is also what the demo's brain inset
should highlight.

Groups come from the dataset's own annotations (superclass, class and cell type), because the
per-neuron neuropil breakdown is not in the public flat files. Silencing means clamping those
neurons' firing rate to zero at every tick, so they neither respond nor pass anything on.

    python -m flybrain.ablate --weights data/train_gpu/model_step11600.pt --n 2048

Writes data/ablation.json.
"""
import argparse
import gzip
import json
import time
import numpy as np
import pandas as pd
import torch
import chess
from .graph import load, DATA
from .eye import Eye
from .policy import FlyPolicy, variant_of, load_into, legal_mask, move_index
from .progress import write as progress

# Functional groups, from the dataset's own annotations. Neuropil membership is not in the public
# flat files (it lives in the neuprint database), and these groups answer the question better
# anyway: "the mushroom body" is Kenyon cells plus their outputs and dopaminergic inputs, which is
# a circuit, not a box. Groups deliberately overlap; each is reported on its own.
GROUPS = {
    # coarse anatomy
    "optic lobe (intrinsic)":      {"superclass": ["ol_intrinsic"]},
    "central brain (intrinsic)":   {"superclass": ["cb_intrinsic"]},
    "visual projection neurons":   {"superclass": ["visual_projection"]},
    "visual centrifugal":          {"superclass": ["visual_centrifugal"]},
    "descending neurons":          {"superclass": ["descending_neuron"]},
    "ascending neurons":           {"superclass": ["ascending_neuron"]},
    "photoreceptors":              {"superclass": ["ol_sensory"]},
    "brain sensory axons":         {"superclass": ["cb_sensory"]},
    # circuits the plan named as candidates
    "mushroom body: Kenyon cells": {"class": ["Kenyon_Cell"]},
    "mushroom body: outputs":      {"class": ["MBON"]},
    "mushroom body: dopaminergic": {"class": ["DAN"]},
    "central complex":             {"class": ["CX"]},
    "lateral horn":                {"type": r"^LH"},
    # senses
    "olfactory receptor neurons":  {"type": r"^ORN"},
    "antennal lobe projection":    {"class": ["ALPN"]},
    "antennal lobe local":         {"class": ["ALLN"]},
    "gustatory":                   {"class": ["gustatory"]},
    "mechanosensory":              {"class": ["mechanosensory", "mechanosensory_tactile", "mechanosensory_proprioceptive"]},
    # the visual pathway, stage by stage
    "lamina (L1-L5)":              {"type": r"^L[1-5]$"},
    "medulla intrinsic (Mi)":      {"type": r"^Mi\d"},
    "distal medulla (Dm)":         {"type": r"^Dm\d"},
    "transmedullary (Tm, TmY)":    {"type": r"^Tm"},
    "motion detectors (T4, T5)":   {"type": r"^T[45][a-d]?$"},
    "lobula plate tangential":     {"type": r"^(HS|VS|H1|H2|CH|VST|VSm)"},
    "lobula columnar (LC, LPLC)":  {"type": r"^(LC|LPLC|LLPC|LPC)\d"},
    "medulla tangential (Pm, Li)": {"type": r"^(Pm|Li)\d"},
    "the wide-field cell CT1":     {"type": r"^CT1$"},
    # the central complex, broken apart
    "CX: ring neurons (ER)":       {"type": r"^ER\d"},
    "CX: compass (EPG, PEG, PEN)": {"type": r"^(EPG|PEG|PEN)"},
    "CX: fan-shaped body":         {"type": r"^(FB|FC|FS|FR|PFN|PFL|PFG|hDelta|vDelta)"},
    "CX: protocerebral bridge":    {"type": r"^(PB|Delta7)"},
}


def group_masks(meta):
    """One boolean mask per functional group."""
    ty = meta.type.fillna("")
    out = {}
    for name, spec in GROUPS.items():
        m = np.zeros(len(meta), bool)
        if "superclass" in spec:
            m |= meta.superclass.isin(spec["superclass"]).to_numpy()
        if "class" in spec:
            m |= meta["class"].isin(spec["class"]).to_numpy()
        if "type" in spec:
            m |= ty.str.match(spec["type"]).to_numpy()
        out[name] = m
    return out


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
    masks = {k: v for k, v in group_masks(meta).items() if v.sum() >= min_neurons}
    targets = list(masks)
    print(f"{len(targets)} functional groups with at least {min_neurons} neurons", flush=True)
    for k in targets: print(f"    {k:30s} {int(masks[k].sum()):6d} neurons", flush=True)
    recs = [json.loads(l) for _, l in zip(range(n_positions), gzip.open(positions, "rt"))]

    base_top1, base_legal = score(model, eye, recs, batch, None, device)
    print(f"intact: top1 {base_top1:.4f}  legal {base_legal:.4f}", flush=True)
    out = {"weights": weights, "n_positions": len(recs), "intact": {"top1": base_top1, "legal": base_legal}, "groups": {}}
    t0 = time.time()
    for i, r in enumerate(targets, 1):
        sil = masks[r]
        t1, lg = score(model, eye, recs, batch, sil, device)
        out["groups"][r] = {"neurons": int(sil.sum()), "top1": t1, "legal": lg,
                             "top1_drop": base_top1 - t1, "legal_drop": base_legal - lg}
        progress(i, len(targets), t0, "ablation")
        print(f"  {r:22s} {int(sil.sum()):6d} neurons silenced -> top1 {t1:.4f} ({t1-base_top1:+.4f})  legal {lg:.4f} ({lg-base_legal:+.4f})", flush=True)
        json.dump(out, open(DATA / "ablation.json", "w"), indent=1)
    worst = sorted(out["groups"].items(), key=lambda kv: -kv[1]["top1_drop"])[:8]
    print("\nmost load-bearing groups:")
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
