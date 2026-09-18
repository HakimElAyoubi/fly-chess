"""Is the fly's chess ability limited by its brain, or by the width of its output?

Phase 2 found that the board is 91.6% readable from the medulla but only 47% readable from the
1,314 descending neurons. If the same narrowing applies to the move, then move quality is capped
by the output pathway, and no amount of training or better labels can lift it.

This asks the question directly. The brain is frozen. Each candidate population is randomly
projected down to exactly 1,314 dimensions, the same width as the descending neurons, so every
population gets an identically sized readout. Then one linear head per population is fit to
Stockfish's move. Differences are therefore about where the information is, not about how many
parameters the readout was given.

    python -m flybrain.bottleneck --weights data/train_gpu/model_step11600.pt --n 40000
"""
import argparse
import gzip
import json
import time
import numpy as np
import torch
import chess
from .graph import load, DATA
from .eye import Eye
from .policy import FlyPolicy, variant_of, load_into, move_index, N_MOVES
from .progress import write as progress

POPULATIONS = {
    "descending_neuron": ["descending_neuron"],
    "central_brain": ["cb_intrinsic"],
    "visual_projection": ["visual_projection"],
    "optic_lobe": ["ol_intrinsic"],
    "descending + central_brain": ["descending_neuron", "cb_intrinsic"],
    "whole_brain": None,
}


def fit_head(Z, y, n_train, epochs=60, lr=3e-3, wd=1e-4, device="cpu"):
    """Linear softmax from Z to the move, trained on the first n_train rows, scored on the rest."""
    Zt = torch.as_tensor(Z, device=device); yt = torch.as_tensor(y, device=device)
    lin = torch.nn.Linear(Z.shape[1], N_MOVES).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    n = n_train
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(lin(Zt[b]), yt[b])
            loss.backward(); opt.step()
    with torch.no_grad():
        pred = lin(Zt[n:]).argmax(1)
        top3 = lin(Zt[n:]).topk(3, dim=1).indices
        return float((pred == yt[n:]).float().mean()), float((top3 == yt[n:, None]).any(1).float().mean())


def run(weights, positions, n_positions, device, batch, seed=0):
    edge, ck = variant_of(weights, device)
    model = FlyPolicy(device=device, ticks=24, edge_gains=edge)
    load_into(model, ck["model"], weights, strict_report=True)
    model.eval()
    W, meta = load(); eye = Eye(W, meta)
    recs = [json.loads(l) for _, l in zip(range(n_positions), gzip.open(positions, "rt"))]
    y = np.array([move_index(chess.Move.from_uci(r["sf_move"]))[0] for r in recs], dtype=np.int64)
    idx = {name: (np.arange(len(meta)) if sc is None else meta.index[meta.superclass.isin(sc)].to_numpy())
           for name, sc in POPULATIONS.items()}
    print({k: len(v) for k, v in idx.items()}, flush=True)

    # one fixed random projection per population, all to the descending-neuron width
    width = len(idx["descending_neuron"])
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = {k: (None if len(v) == width and k == "descending_neuron"
                else (torch.randn(len(v), width, generator=g) / np.sqrt(len(v))).to(device))
            for k, v in idx.items()}
    feats = {k: np.zeros((len(recs), width), np.float32) for k in idx}
    sel = {k: torch.as_tensor(v, device=device) for k, v in idx.items()}

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(recs), batch):
            chunk = recs[i:i + batch]
            I = torch.from_numpy(np.stack([eye.encode(chess.Board(r["fen"])) for r in chunk], 1)).to(device)
            r_full = model.run_full(I)                                    # [N, B] rates of every neuron
            for k in idx:
                x = r_full[sel[k]].T                                      # [B, |pop|]
                feats[k][i:i + len(chunk)] = (x if proj[k] is None else x @ proj[k]).cpu().numpy()
            progress(i + len(chunk), len(recs), t0, "bottleneck: activations")

    n_train = int(0.9 * len(recs))
    out = {"weights": weights, "n_positions": len(recs), "readout_width": width, "n_train": n_train, "populations": {}}
    for k in idx:
        z = feats[k]
        z = (z - z[:n_train].mean(0)) / (z[:n_train].std(0) + 1e-6)
        t1, t3 = fit_head(z, y, n_train, device=device)
        out["populations"][k] = {"neurons": int(len(idx[k])), "top1": t1, "top3": t3}
        print(f"{k:28s} {len(idx[k]):7d} neurons -> top1 {t1:.3f}  top3 {t3:.3f}", flush=True)
    json.dump(out, open(DATA / "bottleneck.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DATA / "train_gpu" / "model_step11600.pt"))
    ap.add_argument("--positions", default=str(DATA / "positions_all_sf.jsonl.gz"))
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    run(a.weights, a.positions, a.n, a.device, a.batch)
