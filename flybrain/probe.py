"""Phase 2 probe: does the fly's visual system carry the whole board?

Generate positions by random play, paint each onto the right eye, run the untrained digital
fly to steady state, and train a linear readout (kernel ridge regression on one-hot targets,
argmax at test time) from the activity of each stage to the 13-way state of every square and
to the side to move. Stages, from the eye inward:
  lamina (L1-L5, right) -> medulla + lobula (all right optic-lobe intrinsic neurons)
  -> visual projection neurons (right) -> central brain intrinsic -> descending neurons.
"""
import json
import sys
import time
import numpy as np
import torch
import chess
from .graph import load, DATA
from .model import FlyBrain
from .eye import Eye, square_state

N_BOARDS, N_TEST, N_VAL = 3072, 512, 512
T_TICKS = 24
BATCH = 128


def random_positions(n, rng, max_plies=80):
    boards = []
    while len(boards) < n:
        b = chess.Board()
        for _ in range(int(rng.integers(0, max_plies))):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(moves[int(rng.integers(len(moves)))])
        if b.is_game_over():
            continue
        boards.append(b)
    return boards


def stages(meta):
    t = meta.type.fillna("")
    R = meta.side == "R"
    return {
        "lamina_R": meta.index[t.str.match(r"^L[1-5]$") & R].to_numpy(),
        "medulla_lobula_R": meta.index[(meta.superclass == "ol_intrinsic") & R].to_numpy(),
        "visual_projection_R": meta.index[meta.superclass.isin(["visual_projection", "visual_projection_tbc"]) & R].to_numpy(),
        "central_brain": meta.index[meta.superclass == "cb_intrinsic"].to_numpy(),
        "descending": meta.index[meta.superclass == "descending_neuron"].to_numpy(),
    }


def softmax_probe(K, Y_sq, y_turn, n_train, n_val, n_comp=1024, epochs=300):
    """Linear classifier on the activity: PCA (from the train kernel) to n_comp components, then
    one softmax regression head per square (13 classes) and one for the side to move, trained
    with cross-entropy and weight decay chosen on the validation split. PCA is a linear map, so
    the whole readout is linear in the activity."""
    tr = slice(0, n_train); va = slice(n_train, n_train + n_val); te = slice(n_train + n_val, None)
    Ktr = K[tr, tr]
    evals, evecs = np.linalg.eigh(Ktr.astype(np.float64))
    order = np.argsort(-evals)[:min(n_comp, n_train - 1)]
    evals, evecs = np.maximum(evals[order], 1e-8), evecs[:, order]
    proj = evecs / np.sqrt(evals)                                       # [n_train, k]: X_train^T-space projection
    Z = (K[:, tr] @ proj).astype(np.float32)                            # [n, k] principal-component scores
    Z /= Z[tr].std(0, keepdims=True) + 1e-6
    Zt = torch.from_numpy(Z); Yt = torch.from_numpy(Y_sq); Tt = torch.from_numpy(y_turn)
    best = None
    for wd in (1e-4, 1e-3, 1e-2, 1e-1):
        torch.manual_seed(0)
        lin = torch.nn.Linear(Z.shape[1], 64 * 13 + 2)
        opt = torch.optim.Adam(lin.parameters(), lr=3e-3, weight_decay=wd)
        for ep in range(epochs):
            opt.zero_grad()
            out = lin(Zt[tr])
            loss = torch.nn.functional.cross_entropy(out[:, :64 * 13].reshape(-1, 13), Yt[tr].reshape(-1)) + torch.nn.functional.cross_entropy(out[:, 64 * 13:], Tt[tr])
            loss.backward(); opt.step()
        with torch.no_grad():
            acc = float((lin(Zt[va])[:, :64 * 13].reshape(-1, 64, 13).argmax(-1) == Yt[va]).float().mean())
        if best is None or acc > best[0]:
            best = (acc, wd, lin)
    acc_val, wd, lin = best
    with torch.no_grad():
        out = lin(Zt[te])
        pred = out[:, :64 * 13].reshape(-1, 64, 13).argmax(-1).numpy()
        turn_pred = out[:, 64 * 13:].argmax(-1).numpy()
    per_class = {}
    for c in range(13):
        m = Y_sq[te] == c
        per_class[c] = float((pred[m] == c).mean()) if m.any() else None
    return {"square_acc": float((pred == Y_sq[te]).mean()), "turn_acc": float((turn_pred == y_turn[te]).mean()),
            "per_class_recall": per_class, "weight_decay": wd, "val_acc": acc_val, "components": int(Z.shape[1])}


def kernel_ridge_probe(X, Y_sq, y_turn, n_train, n_val):
    """X [n, d] float32; Y_sq [n, 64] classes 0..12; y_turn [n]. Returns accuracies on the test split."""
    X = X.astype(np.float32)
    mu = X[:n_train].mean(0, keepdims=True)
    X = X - mu
    K = X @ X.T                                            # [n, n]
    onehot = np.zeros((len(X), 64 * 13 + 2), np.float32)
    for s in range(64):
        onehot[np.arange(len(X)), s * 13 + Y_sq[:, s]] = 1
    onehot[np.arange(len(X)), 64 * 13 + y_turn] = 1
    tr = slice(0, n_train); va = slice(n_train, n_train + n_val); te = slice(n_train + n_val, None)
    scale = np.trace(K[tr, tr]) / n_train
    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        A = np.linalg.solve(K[tr, tr] + lam * scale * np.eye(n_train, dtype=np.float32), onehot[tr])
        pv = K[va, tr] @ A
        acc = square_acc(pv, Y_sq[va])
        if best is None or acc > best[0]:
            best = (acc, lam, A)
    acc_val, lam, A = best
    pt = K[te, tr] @ A
    per_class = {}
    pred = pt[:, :64 * 13].reshape(len(pt), 64, 13).argmax(-1)
    for c in range(13):
        m = Y_sq[te] == c
        per_class[c] = float((pred[m] == c).mean()) if m.any() else None
    turn_pred = pt[:, 64 * 13:].argmax(-1)
    ridge = {"square_acc": square_acc(pt, Y_sq[te]), "turn_acc": float((turn_pred == y_turn[te]).mean()),
             "per_class_recall": per_class, "lambda": lam, "val_acc": acc_val, "features": int(X.shape[1])}
    res = softmax_probe(K, Y_sq, y_turn, n_train, n_val)
    res["features"] = int(X.shape[1]); res["ridge"] = ridge
    return res


def square_acc(p, Y):
    pred = p[:, :64 * 13].reshape(len(p), 64, 13).argmax(-1)
    return float((pred == Y).mean())


def run(device="cpu", n_boards=N_BOARDS):
    W, meta = load()
    cfg = json.load(open(DATA / "checks.json"))["chosen_config"]
    brain = FlyBrain(W, device=device, **cfg)
    eye = Eye(W, meta)
    print("eye:", json.dumps(eye.stats))
    rng = np.random.default_rng(1)
    boards = random_positions(n_boards, rng)
    Y = np.stack([eye.targets(b)[0] for b in boards]); turn = np.array([eye.targets(b)[1] for b in boards])
    print("positions:", len(boards), "| pieces per board mean", float((Y > 0).sum(1).mean()), "| black to move", int(turn.sum()))
    st = stages(meta)
    keep = np.unique(np.concatenate(list(st.values())))
    pr = np.array(sorted(eye.pr_col))
    feats = np.zeros((len(boards), len(keep)), np.float16)
    inputs = np.zeros((len(boards), len(pr)), np.float16)
    t0 = time.time()
    for i in range(0, len(boards), BATCH):
        I = np.stack([eye.encode(b) for b in boards[i:i + BATCH]], 1)          # [N, B]
        inputs[i:i + BATCH] = I[pr].T
        r, _ = brain.run(torch.from_numpy(I).to(brain.device), T=T_TICKS, record_every=0)
        feats[i:i + BATCH] = r.cpu().numpy()[keep].T.astype(np.float16)
        print(f"  {i + len(boards[i:i + BATCH])}/{len(boards)} boards, {time.time() - t0:.0f}s", flush=True)
    n_test = n_val = min(N_TEST, max(32, len(boards) // 6))
    n_train = len(boards) - n_test - n_val
    pos = {k: np.searchsorted(keep, v) for k, v in st.items()}
    results = {"config": cfg, "eye": eye.stats, "n_boards": len(boards), "n_train": n_train, "ticks": T_TICKS, "stages": {}}
    results["stages"]["photoreceptor_input"] = kernel_ridge_probe(inputs, Y, turn, n_train, n_val)
    print("photoreceptor_input    square acc=%.4f (ridge %.4f)" % (results["stages"]["photoreceptor_input"]["square_acc"], results["stages"]["photoreceptor_input"]["ridge"]["square_acc"]), flush=True)
    for name, p in pos.items():
        X = feats[:, p]
        act = float((np.abs(X.astype(np.float32)) > 1e-3).mean())
        res = kernel_ridge_probe(X, Y, turn, n_train, n_val)
        res["frac_responding"] = act
        results["stages"][name] = res
        print(f"{name:22s} neurons={len(p):6d} responding={act:.1%}  square acc={res['square_acc']:.4f} (ridge {res['ridge']['square_acc']:.4f})  turn acc={res['turn_acc']:.3f}", flush=True)
    json.dump(results, open(DATA / "probe.json", "w"), indent=1)
    return results


if __name__ == "__main__":
    run(device=sys.argv[1] if len(sys.argv) > 1 else "cpu", n_boards=int(sys.argv[2]) if len(sys.argv) > 2 else N_BOARDS)
