"""Phase 3 training: imitation of human moves through the digital fly.

python -m flybrain.train [--positions data/positions_pilot.jsonl.gz] [--steps 2000] [--batch 16]
                         [--ticks 24] [--eval-every 100] [--lr 3e-3] [--tag pilot] [--edge-gains]

Loss = cross-entropy on the played move (from x to, 4,096 classes, unmasked so legality must
be learned) + cross-entropy on the promotion piece + cross-entropy on the game result.
Metrics on held-out positions: legal rate of the unmasked argmax, top-1 and top-3 agreement
with the played move among legal moves, value accuracy. Checkpoints and a log go to
data/train_<tag>/; progress to data/progress.txt.
"""
import argparse
import gzip
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import chess
from .graph import load, DATA
from .eye import Eye
from .policy import FlyPolicy, move_index, legal_mask, N_MOVES
from .progress import write as progress


def load_positions(path, n_eval=512, seed=0):
    recs = [json.loads(l) for l in gzip.open(path, "rt")]
    games = sorted({r["game"] for r in recs})
    rng = np.random.default_rng(seed)
    eval_games = set(rng.choice(games, size=max(1, len(games) // 10), replace=False).tolist())
    train = [r for r in recs if r["game"] not in eval_games]
    held = [r for r in recs if r["game"] in eval_games]
    rng.shuffle(held)
    return train, held[:n_eval]


class Batcher:
    def __init__(self, eye, N, device):
        self.eye, self.N, self.device = eye, N, device
        self.pr = np.array(sorted(eye.pr_col))

    def make(self, recs):
        boards = [chess.Board(r["fen"]) for r in recs]
        I = np.stack([self.eye.encode(b) for b in boards], 1)                    # [N, B]
        mv = [chess.Move.from_uci(r["move"]) for r in recs]
        idx = torch.tensor([move_index(m)[0] for m in mv]); promo = torch.tensor([move_index(m)[1] for m in mv])
        val = torch.tensor([r["result"] + 1 for r in recs])                       # 0 black wins, 1 draw, 2 white wins
        return boards, torch.from_numpy(I).to(self.device), idx.to(self.device), promo.to(self.device), val.to(self.device)


@torch.no_grad()
def evaluate(model, batcher, held, batch):
    model.eval()
    legal = top1 = top3 = vacc = n = 0
    loss_sum = 0.0
    for i in range(0, len(held), batch):
        boards, I, idx, promo, val = batcher.make(held[i:i + batch])
        logits, plog, vlog, _ = model(I)
        loss_sum += float(torch.nn.functional.cross_entropy(logits, idx, reduction="sum"))
        mask = legal_mask(boards, model.device)
        un = logits.argmax(1)
        legal += int((mask[torch.arange(len(boards)), un] == 0).sum())
        masked = logits + mask
        top1 += int((masked.argmax(1) == idx).sum())
        top3 += int((masked.topk(3, dim=1).indices == idx[:, None]).any(1).sum())
        vacc += int((vlog.argmax(1) == val).sum())
        n += len(boards)
    model.train()
    return {"loss": loss_sum / n, "legal_rate": legal / n, "top1": top1 / n, "top3": top3 / n, "value_acc": vacc / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default=str(DATA / "positions_pilot.jsonl.gz"))
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ticks", type=int, default=24); ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--lr-brain", type=float, default=3e-3); ap.add_argument("--lr-bias", type=float, default=1e-6)
    ap.add_argument("--tag", default="pilot"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--edge-gains", action="store_true"); ap.add_argument("--resume", default=None)
    ap.add_argument("--n-eval", type=int, default=512); ap.add_argument("--ddp", action="store_true")
    a = ap.parse_args()
    rank, world = 0, 1
    if a.ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        a.device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"; torch.cuda.set_device(a.device)
    main_rank = rank == 0
    out = DATA / f"train_{a.tag}"; out.mkdir(exist_ok=True)
    torch.manual_seed(0)
    model = FlyPolicy(device=a.device, ticks=a.ticks, edge_gains=a.edge_gains)
    net = model
    W, meta = load(); eye = Eye(W, meta)
    batcher = Batcher(eye, model.N, model.device)
    train, held = load_positions(a.positions, n_eval=a.n_eval)
    if main_rank: print(f"positions: {len(train)} train, {len(held)} held out | params: {sum(p.numel() for p in model.parameters()):,} | world {world}", flush=True)
    if not a.resume:
        boards_c, I_c, *_ = batcher.make([train[i] for i in np.random.default_rng(1).integers(len(train), size=32)])
        model.calibrate(I_c)
        if main_rank: print(f"calibrated the descending-neuron standardisation on 32 positions (scale median {float(model.dn_scale.median()):.3g})", flush=True)
    if a.ddp:
        net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    # the descending-neuron signal is ~1e-5 wide, so a bias shift of that size already moves the
    # readout by a full standard deviation: biases get a learning rate of that order, log-gains
    # (multiplicative) a normal one, and every group is clipped on its own
    heads = [p for n, p in model.named_parameters() if n.startswith(("readout", "value"))]
    gains = [p for n, p in model.named_parameters() if "gain" in n]
    biases = [p for n, p in model.named_parameters() if n == "bias"]
    groups = [{"params": heads, "lr": a.lr, "weight_decay": 1e-4}, {"params": gains, "lr": a.lr_brain, "weight_decay": 0.0},
              {"params": biases, "lr": a.lr_bias, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups)
    step0 = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=a.device); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]
    log = open(out / "log.jsonl", "a") if main_rank else None
    rng = np.random.default_rng(step0 * 1000 + rank)
    t0 = time.time(); run_loss = None
    for step in range(step0 + 1, a.steps + 1):
        recs = [train[i] for i in rng.integers(len(train), size=a.batch)]
        boards, I, idx, promo, val = batcher.make(recs)
        logits, plog, vlog, _ = net(I)
        loss = (torch.nn.functional.cross_entropy(logits, idx) + 0.2 * torch.nn.functional.cross_entropy(plog, promo)
                + 0.2 * torch.nn.functional.cross_entropy(vlog, val))
        opt.zero_grad(); loss.backward()
        for g in groups:
            torch.nn.utils.clip_grad_norm_(g["params"], 5.0)
        opt.step()
        run_loss = float(loss) if run_loss is None else 0.98 * run_loss + 0.02 * float(loss)
        if main_rank: progress(step, a.steps, t0, f"train {a.tag}: loss {run_loss:.3f}")
        if main_rank and (step % a.eval_every == 0 or step == a.steps):
            ev = evaluate(model, batcher, held, a.batch)
            ev.update({"step": step, "train_loss": run_loss, "elapsed_min": (time.time() - t0) / 60})
            log.write(json.dumps(ev) + "\n"); log.flush()
            print(f"step {step:6d}  train loss {run_loss:.3f}  eval loss {ev['loss']:.3f}  legal {ev['legal_rate']:.3f}  top1 {ev['top1']:.3f}  top3 {ev['top3']:.3f}  value {ev['value_acc']:.3f}  ({ev['elapsed_min']:.1f} min)", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(a)}, out / "checkpoint.pt")


if __name__ == "__main__":
    main()
