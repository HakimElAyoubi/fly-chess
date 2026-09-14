"""Refit the Phase 2 linear readouts from the cached features (data/probe_features.npz).

Two linear readouts of each stage's steady-state activity:

1. Generic: PCA from the training kernel (components kept down to a floor on the eigenvalue
   spectrum), one global scale or a floored partial whitening, one softmax head per square
   (13 classes) plus one for the side to move, L-BFGS to convergence; the component floor, the
   scaling and the weight penalty are chosen on the validation split. Only the medulla runs the
   full grid; the other stages reuse its best setting (the photoreceptor input and the
   descending neurons also try a short grid).
2. Retinotopic (stages with eye columns only): each square is read by its own linear head from
   the neurons assigned to that square's columns. This is how a downstream neuron would read
   the medulla, and it is still a linear readout of the activity.

Writes data/probe.json. Progress in data/progress.txt.
"""
import json
import time
import numpy as np
import torch
from .graph import load, DATA
from .probe import stages, N_TEST, N_VAL
from .progress import write as progress
from .eye import Eye

FLOORS = (1e-5, 1e-6, 1e-7)                      # keep components with eigenvalue > floor * top eigenvalue
SCALINGS = {"global": (1e-7, 3e-7), "partial": (1e-8, 1e-7)}
SHORT_GRID = [(1e-5, "global", 1e-7), (1e-6, "global", 1e-7), (1e-5, "partial", 1e-8)]
MAIN_STAGE = "medulla_lobula_R"
NAMES = ["empty", "wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"]


def fit_head(Z, Y_sq, y_turn, tr, wd, max_iter=500, n_out=64 * 13 + 2):
    Zt, Yt, Tt = torch.from_numpy(Z[tr]), torch.from_numpy(Y_sq[tr]), torch.from_numpy(y_turn[tr])
    torch.manual_seed(0)
    lin = torch.nn.Linear(Z.shape[1], n_out)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        out = lin(Zt)
        loss = torch.nn.functional.cross_entropy(out[:, :64 * 13].reshape(-1, 13), Yt.reshape(-1)) + wd * (lin.weight ** 2).sum()
        if n_out > 64 * 13:
            loss = loss + torch.nn.functional.cross_entropy(out[:, 64 * 13:], Tt)
        loss.backward(); return loss
    opt.step(closure)
    return lin


def evaluate(lin, Z, Y_sq, y_turn, sl):
    with torch.no_grad():
        out = lin(torch.from_numpy(Z[sl]))
        pred = out[:, :64 * 13].reshape(-1, 64, 13).argmax(-1).numpy()
        turn_pred = out[:, 64 * 13:].argmax(-1).numpy() if out.shape[1] > 64 * 13 else np.zeros(len(pred), int)
    per_class = {NAMES[c]: (float((pred[Y_sq[sl] == c] == c).mean()) if (Y_sq[sl] == c).any() else None) for c in range(13)}
    return {"square_acc": float((pred == Y_sq[sl]).mean()), "turn_acc": float((turn_pred == y_turn[sl]).mean()), "per_class_recall": per_class}


def probe_stage(X, Y_sq, y_turn, n_train, n_val, test_sets, grid=None, report=None):
    X = X.astype(np.float32)
    X = X - X[:n_train].mean(0, keepdims=True)
    K = X @ X.T
    tr = slice(0, n_train); va = slice(n_train, n_train + n_val)
    evals, evecs = np.linalg.eigh(K[tr, tr].astype(np.float64))
    order = np.argsort(-evals); evals, evecs = np.maximum(evals[order], 1e-12), evecs[:, order]
    if grid is None:
        grid = [(fl, s, wd) for fl in FLOORS for s, wds in SCALINGS.items() for wd in wds]
    best = None
    for floor, scaling, wd in grid:
        k = int((evals > floor * evals[0]).sum()); k = max(8, min(k, n_train - 1))
        Z = (K[:, tr] @ (evecs[:, :k] / np.sqrt(evals[:k]))).astype(np.float32)           # principal-component scores
        if scaling == "partial":
            sd = Z[tr].std(0, keepdims=True); Z = Z / np.sqrt(sd ** 2 + 1e-4 * (sd ** 2).max())
        Z = Z / (np.sqrt((Z[tr] ** 2).mean()) + 1e-9)
        lin = fit_head(Z, Y_sq, y_turn, tr, wd)
        acc = evaluate(lin, Z, Y_sq, y_turn, va)["square_acc"]
        if not np.isfinite(acc) or acc < 0.3:
            acc = 0.0                                                                     # numerical failure, never chosen
        if report: report(floor, k, scaling, wd, acc)
        if best is None or acc > best[0]:
            best = (acc, floor, k, scaling, wd, lin, Z)
    acc_val, floor, k, scaling, wd, lin, Z = best
    res = {"features": int(X.shape[1]), "floor": floor, "components": int(k), "scaling": scaling, "weight_decay": wd, "val_acc": acc_val,
           "spectrum": {"n_above_1e-3": int((evals > 1e-3 * evals[0]).sum()), "n_above_1e-5": int((evals > 1e-5 * evals[0]).sum()), "n_above_1e-7": int((evals > 1e-7 * evals[0]).sum())},
           "frac_responding": float((np.abs(X) > 1e-3).mean())}
    for name, sl in test_sets.items():
        res[name] = evaluate(lin, Z, Y_sq, y_turn, sl)
    return res


def retinotopic_probe(X, neuron_idx, eye, Y_sq, n_train, n_val, test_sets, wd=1e-5):
    """Each square read by a linear head from the neurons whose eye column lies in that square."""
    sq_of = np.full(len(neuron_idx), -1, int)
    for j, i in enumerate(neuron_idx):
        c = eye.col_of.get(int(i))
        if c is not None:
            sq_of[j] = eye.col_square[eye.col_index[c]]
    tr = slice(0, n_train)
    preds = {name: np.zeros((sl.stop - sl.start if sl.stop else len(X) - sl.start, 64), int) for name, sl in test_sets.items()}
    used = []
    for s in range(64):
        cols = np.where(sq_of == s)[0]
        used.append(len(cols))
        if len(cols) == 0:
            continue
        Xs = X[:, cols].astype(np.float32); Xs = Xs / (np.abs(Xs[tr]).mean() + 1e-9)
        Xt, yt = torch.from_numpy(Xs[tr]), torch.from_numpy(Y_sq[tr, s])
        torch.manual_seed(0); lin = torch.nn.Linear(len(cols), 13)
        opt = torch.optim.LBFGS(lin.parameters(), max_iter=300, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); loss = torch.nn.functional.cross_entropy(lin(Xt), yt) + wd * (lin.weight ** 2).sum(); loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            for name, sl in test_sets.items():
                preds[name][:, s] = lin(torch.from_numpy(Xs[sl])).argmax(1).numpy()
    res = {"neurons_per_square_mean": float(np.mean(used)), "neurons_per_square_min": int(np.min(used)), "weight_decay": wd}
    for name, sl in test_sets.items():
        pred = preds[name]; Yt = Y_sq[sl]
        res[name] = {"square_acc": float((pred == Yt).mean()),
                     "per_class_recall": {NAMES[c]: (float((pred[Yt == c] == c).mean()) if (Yt == c).any() else None) for c in range(13)}}
    return res


def run():
    W, meta = load()
    eye = Eye(W, meta)
    cache = np.load(DATA / "probe_features.npz")
    feats, inputs, keep, Y, turn, n_play = cache["feats"], cache["inputs"], cache["keep"], cache["Y"], cache["turn"], int(cache["n_play"])
    n = len(Y); n_rand = n - n_play
    n_test = n_val = min(N_TEST, max(32, n_rand // 6)); n_train = n_rand - n_test - n_val
    test_sets = {"test_random_placements": slice(n_train + n_val, n_rand), "test_random_play": slice(n_rand, None)}
    st = stages(meta); pos = {k: np.searchsorted(keep, v) for k, v in st.items()}
    cfg = json.load(open(DATA / "checks.json"))["chosen_config"]
    results = {"config": cfg, "n_boards": int(n), "n_train": n_train, "n_val": n_val, "n_test_random": n_test, "n_test_random_play": n_play,
               "ticks": 24, "floors_tried": FLOORS, "scalings_tried": SCALINGS, "stages": {}}
    order = [MAIN_STAGE, "photoreceptor_input", "descending", "lamina_R", "visual_projection_R", "central_brain"]
    full = len(FLOORS) * sum(len(w) for w in SCALINGS.values())
    total = full + 2 * len(SHORT_GRID) + 3 + 2; done = [0]
    t0 = time.time(); progress(0, total, t0, "probe: fitting readouts")
    best_cfg = None
    for name in order:
        X = inputs if name == "photoreceptor_input" else feats[:, pos[name]]
        def report(fl, k, s, wd, acc):
            done[0] += 1; progress(done[0], total, t0, "probe: fitting readouts")
            print(f"    {name}: floor={fl:g} k={k} {s} wd={wd:g} -> val {acc:.4f}", flush=True)
        grid = None if name == MAIN_STAGE else SHORT_GRID if name in ("photoreceptor_input", "descending") else [best_cfg]
        res = probe_stage(X, Y, turn, n_train, n_val, test_sets, grid=grid, report=report)
        if name == MAIN_STAGE:
            best_cfg = (res["floor"], res["scaling"], res["weight_decay"])
        if name in ("lamina_R", MAIN_STAGE):
            res["retinotopic"] = retinotopic_probe(X, st[name], eye, Y, n_train, n_val, test_sets)
            done[0] += 1; progress(done[0], total, t0, "probe: fitting readouts")
        results["stages"][name] = res
        extra = f"  retinotopic: random {res['retinotopic']['test_random_placements']['square_acc']:.4f} play {res['retinotopic']['test_random_play']['square_acc']:.4f}" if "retinotopic" in res else ""
        print(f"{name:22s} features={res['features']:6d} responding={res['frac_responding']:.1%}  k={res['components']:4d} {res['scaling']} wd={res['weight_decay']:g}"
              f"  square acc: random {res['test_random_placements']['square_acc']:.4f}  play {res['test_random_play']['square_acc']:.4f}  turn {res['test_random_placements']['turn_acc']:.3f}{extra}", flush=True)
        json.dump(results, open(DATA / "probe.json", "w"), indent=1)
    progress(total, total, t0, "probe: fitting readouts")


if __name__ == "__main__":
    run()
