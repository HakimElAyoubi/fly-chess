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
from .progress import write as progress

N_BOARDS, N_TEST, N_VAL = 5120, 512, 512
T_TICKS = 24
BATCH = 128


def random_placements(n, rng, p_empty=0.4):
    """Uniformly random boards: each square empty with p_empty, else a uniform piece. Not legal
    chess, deliberately: every piece appears on every square often, so the readout of every
    (square, state) pair is trained on hundreds of examples."""
    boards = []
    for _ in range(n):
        b = chess.Board(None)
        for sq in chess.SQUARES:
            if rng.random() >= p_empty:
                b.set_piece_at(sq, chess.Piece(int(rng.integers(1, 7)), bool(rng.integers(2))))
        b.turn = bool(rng.integers(2))
        boards.append(b)
    return boards


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


def softmax_probe(K, Y_sq, y_turn, n_train, n_val, n_comp=2048, test_sets=None):
    """Linear classifier on the activity: PCA (from the train kernel) to at most n_comp components,
    then one softmax regression head per square (13 classes) and one for the side to move, fit
    to convergence with L-BFGS; the weight penalty is chosen on the validation split. PCA is a
    linear map, so the whole readout is linear in the activity."""
    tr = slice(0, n_train); va = slice(n_train, n_train + n_val)
    Ktr = K[tr, tr]
    evals, evecs = np.linalg.eigh(Ktr.astype(np.float64))
    order = np.argsort(-evals)
    order = order[evals[order] > 1e-7 * evals[order[0]]][:min(n_comp, n_train - 1)]
    evals, evecs = evals[order], evecs[:, order]
    proj = evecs / np.sqrt(evals)                                       # [n_train, k]
    Z = (K[:, tr] @ proj).astype(np.float32)                            # [n, k] principal-component scores
    Z /= np.sqrt((Z[tr] ** 2).mean()) + 1e-9                            # one global scale, no per-component whitening
    Zt = torch.from_numpy(Z); Yt = torch.from_numpy(Y_sq); Tt = torch.from_numpy(y_turn)

    def fit(wd):
        torch.manual_seed(0)
        lin = torch.nn.Linear(Z.shape[1], 64 * 13 + 2)
        opt = torch.optim.LBFGS(lin.parameters(), max_iter=400, history_size=20, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            out = lin(Zt[tr])
            loss = (torch.nn.functional.cross_entropy(out[:, :64 * 13].reshape(-1, 13), Yt[tr].reshape(-1))
                    + torch.nn.functional.cross_entropy(out[:, 64 * 13:], Tt[tr]) + wd * (lin.weight ** 2).sum())
            loss.backward(); return loss
        opt.step(closure)
        return lin

    def evaluate(lin, sl):
        with torch.no_grad():
            out = lin(Zt[sl])
            pred = out[:, :64 * 13].reshape(-1, 64, 13).argmax(-1).numpy()
            turn_pred = out[:, 64 * 13:].argmax(-1).numpy()
        per_class = {}
        for c in range(13):
            m = Y_sq[sl] == c
            per_class[c] = float((pred[m] == c).mean()) if m.any() else None
        return {"square_acc": float((pred == Y_sq[sl]).mean()), "turn_acc": float((turn_pred == y_turn[sl]).mean()), "per_class_recall": per_class}

    best = None
    for wd in (1e-6, 1e-5, 1e-4, 1e-3):
        lin = fit(wd)
        acc = evaluate(lin, va)["square_acc"]
        if best is None or acc > best[0]:
            best = (acc, wd, lin)
    acc_val, wd, lin = best
    res = {"weight_decay": wd, "val_acc": acc_val, "components": int(Z.shape[1])}
    for name, sl in (test_sets or {"test": slice(n_train + n_val, None)}).items():
        res[name] = evaluate(lin, sl)
    res["square_acc"] = res[next(iter(test_sets or {"test": 0}))]["square_acc"]
    res["turn_acc"] = res[next(iter(test_sets or {"test": 0}))]["turn_acc"]
    return res


def kernel_ridge_probe(X, Y_sq, y_turn, n_train, n_val, test_sets=None):
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
    res = softmax_probe(K, Y_sq, y_turn, n_train, n_val, test_sets=test_sets)
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
    n_play = min(1024, n_boards // 5)
    boards = random_placements(n_boards - n_play, rng) + random_positions(n_play, rng)   # random-play boards last: a realistic test set
    Y = np.stack([eye.targets(b)[0] for b in boards]); turn = np.array([eye.targets(b)[1] for b in boards])
    print("positions:", len(boards), f"({n_boards - n_play} random placements + {n_play} random-play)", "| pieces per board mean", float((Y > 0).sum(1).mean()), "| black to move", int(turn.sum()))
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
        print("  " + progress(i + len(boards[i:i + BATCH]), len(boards), t0, "probe: simulating"), flush=True)
    np.savez_compressed(DATA / "probe_features.npz", feats=feats, inputs=inputs, keep=keep, pr=pr, Y=Y, turn=turn, n_play=n_play)
    if "nofit" in sys.argv:
        print("features cached; run python -m flybrain.probe_fit", flush=True); return
    n_rand = len(boards) - n_play
    n_test = n_val = min(N_TEST, max(32, n_rand // 6))
    n_train = n_rand - n_test - n_val
    test_sets = {"test_random_placements": slice(n_train + n_val, n_rand), "test_random_play": slice(n_rand, None)}
    pos = {k: np.searchsorted(keep, v) for k, v in st.items()}
    results = {"config": cfg, "eye": eye.stats, "n_boards": len(boards), "n_train": n_train, "n_random_play_test": n_play, "ticks": T_TICKS, "stages": {}}
    results["stages"]["photoreceptor_input"] = kernel_ridge_probe(inputs, Y, turn, n_train, n_val, test_sets)
    r0 = results["stages"]["photoreceptor_input"]
    print("photoreceptor_input    square acc: random %.4f  play %.4f (ridge %.4f)" % (r0["test_random_placements"]["square_acc"], r0["test_random_play"]["square_acc"], r0["ridge"]["square_acc"]), flush=True)
    for name, p in pos.items():
        X = feats[:, p]
        act = float((np.abs(X.astype(np.float32)) > 1e-3).mean())
        res = kernel_ridge_probe(X, Y, turn, n_train, n_val, test_sets)
        res["frac_responding"] = act
        results["stages"][name] = res
        print(f"{name:22s} neurons={len(p):6d} responding={act:.1%}  square acc: random {res['test_random_placements']['square_acc']:.4f}  play {res['test_random_play']['square_acc']:.4f}  (ridge {res['ridge']['square_acc']:.4f})  turn acc={res['turn_acc']:.3f}", flush=True)
    json.dump(results, open(DATA / "probe.json", "w"), indent=1)
    return results


if __name__ == "__main__":
    run(device=sys.argv[1] if len(sys.argv) > 1 else "cpu", n_boards=int(sys.argv[2]) if len(sys.argv) > 2 else N_BOARDS)
