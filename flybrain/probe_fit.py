"""Refit the Phase 2 linear readouts from the cached features (data/probe_features.npz).

For each stage: centre, linear kernel, PCA from the training kernel; the number of components
and the weight penalty are chosen together on the validation split; one softmax regression
head per square (13 classes) plus one for the side to move, fit with L-BFGS. Writes
data/probe.json. Progress in data/progress.txt.
"""
import json
import sys
import time
import numpy as np
import torch
from .graph import load, DATA
from .probe import stages, N_TEST, N_VAL, random_placements, random_positions
from .progress import write as progress

COMPONENTS = (256, 512, 1024)
WEIGHT_DECAY = (1e-6, 1e-5)
NAMES = ["empty", "wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"]


def fit_head(Z, Y_sq, y_turn, tr, wd, max_iter=400):
    Zt, Yt, Tt = torch.from_numpy(Z[tr]), torch.from_numpy(Y_sq[tr]), torch.from_numpy(y_turn[tr])
    torch.manual_seed(0)
    lin = torch.nn.Linear(Z.shape[1], 64 * 13 + 2)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        out = lin(Zt)
        loss = (torch.nn.functional.cross_entropy(out[:, :64 * 13].reshape(-1, 13), Yt.reshape(-1))
                + torch.nn.functional.cross_entropy(out[:, 64 * 13:], Tt) + wd * (lin.weight ** 2).sum())
        loss.backward(); return loss
    opt.step(closure)
    return lin


def evaluate(lin, Z, Y_sq, y_turn, sl):
    with torch.no_grad():
        out = lin(torch.from_numpy(Z[sl]))
        pred = out[:, :64 * 13].reshape(-1, 64, 13).argmax(-1).numpy()
        turn_pred = out[:, 64 * 13:].argmax(-1).numpy()
    per_class = {NAMES[c]: (float((pred[Y_sq[sl] == c] == c).mean()) if (Y_sq[sl] == c).any() else None) for c in range(13)}
    return {"square_acc": float((pred == Y_sq[sl]).mean()), "turn_acc": float((turn_pred == y_turn[sl]).mean()), "per_class_recall": per_class}


def probe_stage(X, Y_sq, y_turn, n_train, n_val, test_sets):
    X = X.astype(np.float32)
    X = X - X[:n_train].mean(0, keepdims=True)
    K = X @ X.T
    tr = slice(0, n_train); va = slice(n_train, n_train + n_val)
    evals, evecs = np.linalg.eigh(K[tr, tr].astype(np.float64))
    order = np.argsort(-evals); evals, evecs = np.maximum(evals[order], 1e-12), evecs[:, order]
    best = None
    for k in COMPONENTS:
        k = min(k, n_train - 1)
        Z = (K[:, tr] @ (evecs[:, :k] / np.sqrt(evals[:k]))).astype(np.float32)
        Z /= np.sqrt((Z[tr] ** 2).mean()) + 1e-9
        for wd in WEIGHT_DECAY:
            lin = fit_head(Z, Y_sq, y_turn, tr, wd)
            acc = evaluate(lin, Z, Y_sq, y_turn, va)["square_acc"]
            if best is None or acc > best[0]:
                best = (acc, k, wd, lin, Z)
    acc_val, k, wd, lin, Z = best
    res = {"features": int(X.shape[1]), "components": int(k), "weight_decay": wd, "val_acc": acc_val,
           "frac_responding": float((np.abs(X) > 1e-3).mean())}
    for name, sl in test_sets.items():
        res[name] = evaluate(lin, Z, Y_sq, y_turn, sl)
    return res


def run():
    W, meta = load()
    cache = np.load(DATA / "probe_features.npz")
    feats, inputs, keep, Y, turn, n_play = cache["feats"], cache["inputs"], cache["keep"], cache["Y"], cache["turn"], int(cache["n_play"])
    n = len(Y); n_rand = n - n_play
    n_test = n_val = min(N_TEST, max(32, n_rand // 6)); n_train = n_rand - n_test - n_val
    test_sets = {"test_random_placements": slice(n_train + n_val, n_rand), "test_random_play": slice(n_rand, None)}
    st = stages(meta); pos = {k: np.searchsorted(keep, v) for k, v in st.items()}
    cfg = json.load(open(DATA / "checks.json"))["chosen_config"]
    results = {"config": cfg, "n_boards": int(n), "n_train": n_train, "n_val": n_val, "n_test_random": n_test, "n_test_random_play": n_play,
               "ticks": 24, "components_tried": COMPONENTS, "weight_decays_tried": WEIGHT_DECAY, "stages": {}}
    order = [("photoreceptor_input", inputs)] + [(k, feats[:, p]) for k, p in pos.items()]
    t0 = time.time()
    for i, (name, X) in enumerate(order):
        progress(i, len(order), t0, "probe: fitting readouts")
        res = probe_stage(X, Y, turn, n_train, n_val, test_sets)
        results["stages"][name] = res
        print(f"{name:22s} features={res['features']:6d} responding={res['frac_responding']:.1%}  comps={res['components']:4d} wd={res['weight_decay']:g}"
              f"  square acc: random {res['test_random_placements']['square_acc']:.4f}  play {res['test_random_play']['square_acc']:.4f}  turn {res['test_random_placements']['turn_acc']:.3f}", flush=True)
        json.dump(results, open(DATA / "probe.json", "w"), indent=1)
    progress(len(order), len(order), t0, "probe: fitting readouts")


if __name__ == "__main__":
    run()
