"""Phase 7: reinforcement learning. The fly plays, and learns from how the games end.

    python -m flybrain.rl --weights data/train_gpu/model_step11600.pt --opponent greedy \
                          --iters 200 --envs 32 --horizon 8 --device cuda

Everything before this was imitation. The fly was shown a position and told which move someone
else had played, and it was scored on whether it matched. It was never once told that a move it
chose had lost a rook. This trains it on that instead: it plays its own games, and the gradient
comes from the result.

The algorithm is PPO (Schulman et al. 2017), the standard choice for this shape of problem:

  rollout    `envs` games run in parallel. At every decision point the position goes through the
             connectome, the move head is masked to legal moves, and a move is *sampled* from
             that distribution rather than taken greedily, because a policy that never varies
             can never discover anything. The log-probability of the move it chose, the critic's
             estimate of the position, and the reward are recorded.
  advantage  GAE(lambda) (Schulman et al. 2015) turns the rewards into an advantage per move:
             how much better the game went than the critic expected. This is what makes credit
             assignment possible over a hundred-ply game.
  update     several passes over the collected moves, each maximising the clipped surrogate
             objective, so that no single batch of games can move the policy far. Plus a value
             loss for the critic, an entropy bonus against premature collapse, and a penalty on
             the KL divergence from the supervised policy, which is the anchor that stops the
             fly from unlearning the chess it already knows in pursuit of a shaping reward.

What is learned: by default the per-neuron gains and biases and the move head. The wiring, the
signs and the synapse counts stay fixed, as they have in every phase. `--train all` also frees
the 21 M per-edge gains; `--train heads` freezes the brain entirely and moves only the readout.

The critic is a new scalar head on the same descending neurons the move head reads, trained
from scratch here: the supervised value head predicts a three-way game result from White's
point of view, which is not the baseline this needs.

Writes data/rl_<tag>/{checkpoint.pt, best.pt, log.jsonl} and a progress bar to data/progress.txt.
"""
import argparse
import json
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import chess
from .graph import load, DATA
from .eye import Eye
from .env import BatchChessEnv
from .policy import FlyPolicy, load_into, variant_of, legal_mask, N_MOVES, PROMO_INV
from .progress import write as progress

NEG = -1e9                     # a finite stand-in for -inf, so entropy and log-softmax stay finite


def encode(eye, boards, device):
    return torch.from_numpy(np.stack([eye.encode(b) for b in boards], 1)).to(device)


def mask_of(boards, device):
    return legal_mask(boards, device).clamp_(min=NEG)


def resolve(idx, promo_logits, boards):
    """Turn a from-to index into a legal move, asking the promotion head when a pawn promotes."""
    promo = promo_logits[:, 1:].argmax(1) + 1
    out = []
    for b, i, p in zip(boards, idx.tolist(), promo.tolist()):
        m = chess.Move(i // 64, i % 64)
        if m not in b.legal_moves:
            m = chess.Move(i // 64, i % 64, promotion=PROMO_INV[p])
            if m not in b.legal_moves:
                m = chess.Move(i // 64, i % 64, promotion=chess.QUEEN)
        out.append(m)
    return out


class Critic(nn.Module):
    """One number per position: how good this is for the fly, read off the descending neurons.
    Zero-initialised, so it starts as a baseline of 0 and cannot inject noise into the first
    advantages it is asked to explain."""
    def __init__(self, width, device):
        super().__init__()
        self.head = nn.Linear(width, 1).to(device)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, dn):
        return self.head(dn).squeeze(-1)


def gae(rew, val, done, last_val, gamma, lam):
    """Generalised advantage estimation over a [T, n_envs] rollout. `done` marks the step at which
    an episode ended; the bootstrap is cut there because the next row is a different game."""
    T = len(rew)
    adv = np.zeros_like(rew)
    nxt = last_val
    run = np.zeros_like(last_val)
    for t in range(T - 1, -1, -1):
        alive = 1.0 - done[t]
        delta = rew[t] + gamma * nxt * alive - val[t]
        run = delta + gamma * lam * alive * run
        adv[t] = run
        nxt = val[t]
    return adv, adv + val


@torch.no_grad()
def play_out(model, eye, a, games, temperature=0.0):
    """Play `games` complete games against the evaluation opponent and return the score."""
    env = BatchChessEnv(n=min(a.envs, games), opponent=a.eval_opponent or a.opponent, seed=a.seed + 777,
                        max_plies=a.max_plies, gamma=a.gamma, shaping=0.0,
                        stockfish=a.stockfish, sf_depth=a.sf_depth, sf_elo=a.sf_elo)
    while len(env.finished) < games:
        boards = env.observe()
        logits, plog, _, _ = model(encode(eye, boards, model.device))
        logits = logits + mask_of(boards, model.device)
        idx = (torch.multinomial(torch.softmax(logits / temperature, 1), 1).squeeze(1)
               if temperature > 0 else logits.argmax(1))
        env.step(resolve(idx, plog, boards))
    env.close()
    res = np.array([f[0] for f in env.finished[:games]])
    return {"games": int(len(res)), "score": float(res.mean()),
            "wins": int((res == 1).sum()), "draws": int((res == 0.5).sum()), "losses": int((res == 0).sum()),
            "mean_plies": float(np.mean([f[1] for f in env.finished[:games]]))}


def main():
    a = parse_args()
    out = DATA / f"rl_{a.tag}"; out.mkdir(exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device(a.device)

    edge, ck = variant_of(a.weights, a.device)
    model = FlyPolicy(device=a.device, ticks=a.ticks, edge_gains=edge)
    load_into(model, ck["model"], a.weights)
    W, meta = load(); eye = Eye(W, meta)
    critic = Critic(len(model.dn), a.device)
    # the supervised policy, kept frozen as the anchor for the KL penalty
    ref_state = {k: v.detach().clone() for k, v in model.state_dict().items()} if a.kl_ref_coef > 0 else None

    trainable = {"heads": ("readout",), "gains": ("readout", "log_gain", "bias"), "all": ("readout", "log_gain", "bias", "log_edge_gain")}[a.train]
    for n, p in model.named_parameters():
        p.requires_grad_(any(n.startswith(t) for t in trainable))
    groups = [
        {"params": [p for n, p in model.named_parameters() if n.startswith("readout")], "lr": a.lr_head, "weight_decay": 1e-4},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "gain" in n], "lr": a.lr_gain, "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n == "bias"], "lr": a.lr_bias, "weight_decay": 0.0},
        {"params": list(critic.parameters()), "lr": a.lr_critic, "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups)
    if a.resume:
        rc = torch.load(a.resume, map_location=a.device, weights_only=False)
        load_into(model, rc["model"], a.resume); critic.load_state_dict(rc["critic"]); opt.load_state_dict(rc["opt"])
        print(f"resumed from {a.resume} at iteration {rc['iter']}", flush=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in critic.parameters())
    print(f"policy {sum(p.numel() for p in model.parameters()):,} params, training {n_train:,} "
          f"({a.train}) | opponent {a.opponent} | {a.envs} games in parallel", flush=True)

    env = BatchChessEnv(n=a.envs, opponent=a.opponent, opening_plies=a.opening_plies, max_plies=a.max_plies,
                        seed=a.seed, gamma=a.gamma, shaping=a.shaping,
                        stockfish=a.stockfish, sf_depth=a.sf_depth, sf_elo=a.sf_elo)
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    recent = deque(maxlen=200)                  # results of the last games played, for the running score
    best = -1.0
    history = []

    for it in range(1, a.iters + 1):
        # ---------------------------------------------------------------- rollout
        fens, acts, logps, vals, rews, dones, refs = [], [], [], [], [], [], []
        model.eval()
        n_done0 = len(env.finished)
        with torch.no_grad():
            for t in range(a.horizon):
                boards = env.observe()
                I = encode(eye, boards, device)
                mask = mask_of(boards, device)                               # legal moves in *this* position
                logits, plog, _, dn = model(I)
                logp_all = torch.log_softmax((logits + mask) / a.temperature, 1)
                idx = torch.multinomial(logp_all.exp(), 1).squeeze(1)
                if ref_state is not None:                                    # the supervised policy on the same position
                    ref_logits, _, _, _ = torch.func.functional_call(model, ref_state, (I,))
                    refs.append(torch.log_softmax((ref_logits + mask) / a.temperature, 1)
                                .clamp(min=-30).half().cpu())                # -inf is not representable in half
                fens.append([b.fen() for b in boards])                       # before the step: step mutates the boards
                acts.append(idx.cpu().numpy()); logps.append(logp_all.gather(1, idx[:, None]).squeeze(1).cpu().numpy())
                vals.append(critic(dn).cpu().numpy())
                r, d = env.step(resolve(idx, plog, boards))
                rews.append(r); dones.append(d.astype(np.float32))
            last_val = critic(model(encode(eye, env.observe(), device))[3]).cpu().numpy()

        rew = np.array(rews, np.float32); val = np.array(vals, np.float32); dn_mask = np.array(dones, np.float32)
        adv, ret = gae(rew, val, dn_mask, last_val, a.gamma, a.lam)
        flat_fen = [f for row in fens for f in row]
        flat = lambda x: torch.as_tensor(np.concatenate(x), device=device)
        A = torch.as_tensor(adv.reshape(-1), device=device); R = torch.as_tensor(ret.reshape(-1), device=device)
        A = (A - A.mean()) / (A.std() + 1e-6)
        act = flat(acts).long(); old_logp = flat(logps)
        ref_logp = torch.cat(refs, 0).to(device).float() if refs else None
        recent.extend(f[0] for f in env.finished[n_done0:])

        # ---------------------------------------------------------------- PPO update
        model.train()
        n = len(flat_fen); stats = []
        stop = False
        for ep in range(a.epochs):
            order = np.random.permutation(n)
            for s in range(0, n, a.minibatch):
                b = order[s:s + a.minibatch]
                boards = [chess.Board(flat_fen[i]) for i in b]
                logits, _, _, dnf = model(encode(eye, boards, device))
                masked = logits + mask_of(boards, device)
                logp_all = torch.log_softmax(masked / a.temperature, 1)
                bi = torch.as_tensor(b, device=device)
                logp = logp_all.gather(1, act[bi][:, None]).squeeze(1)
                ratio = (logp - old_logp[bi]).exp()
                adv_b = A[bi]
                pg = -torch.min(ratio * adv_b, ratio.clamp(1 - a.clip, 1 + a.clip) * adv_b).mean()
                v = critic(dnf)
                vf = 0.5 * (v - R[bi]).pow(2).mean()
                p = logp_all.exp()
                ent = -(p * logp_all.clamp(min=NEG)).sum(1).mean()
                if ref_logp is not None:
                    pr = ref_logp[bi].exp()
                    kl_ref = torch.where(pr > 0, pr * (ref_logp[bi] - logp_all.clamp(min=-30)),
                                         torch.zeros_like(pr)).sum(1).mean()
                else:
                    kl_ref = torch.zeros((), device=device)
                loss = pg + a.vf_coef * vf - a.ent_coef * ent + a.kl_ref_coef * kl_ref
                opt.zero_grad(); loss.backward()
                for g in groups:
                    torch.nn.utils.clip_grad_norm_(g["params"], a.max_grad_norm)
                opt.step()
                approx_kl = float((old_logp[bi] - logp).mean().detach())
                stats.append({"pg": float(pg), "vf": float(vf), "entropy": float(ent),
                              "kl_ref": float(kl_ref), "kl_old": approx_kl,
                              "clipped": float(((ratio - 1).abs() > a.clip).float().mean())})
                if a.target_kl and abs(approx_kl) > a.target_kl:              # the batch has moved the policy far enough
                    stop = True; break
            if stop:
                break

        # ---------------------------------------------------------------- report
        m = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
        row = {"iter": it, "steps": it * a.horizon * a.envs, "games_finished": len(env.finished),
               "reward_mean": float(rew.mean()), "return_mean": float(ret.mean()),
               "value_mean": float(val.mean()), "score_recent": float(np.mean(recent)) if recent else None,
               "updates": len(stats), "early_stop": stop, "elapsed_min": (time.time() - t0) / 60, **m}
        history.append(row); log.write(json.dumps(row) + "\n"); log.flush()
        progress(it, a.iters, t0, f"rl {a.tag}: score {row['score_recent'] or 0:.3f}")
        print(f"iter {it:4d}  reward {row['reward_mean']:+.4f}  score(last {len(recent)}) "
              f"{row['score_recent'] or float('nan'):.3f}  entropy {m['entropy']:.2f}  klref {m['kl_ref']:.4f}  "
              f"klold {m['kl_old']:+.4f}  clip {m['clipped']:.2f}  ({row['elapsed_min']:.1f} min)", flush=True)

        out_of_time = a.hours > 0 and (time.time() - t0) / 3600 >= a.hours
        if it % a.eval_every == 0 or it == a.iters or out_of_time:
            ev = play_out(model, eye, a, a.eval_games)
            ev.update({"iter": it, "kind": "eval", "opponent": a.eval_opponent or a.opponent})
            log.write(json.dumps(ev) + "\n"); log.flush()
            print(f"  eval vs {ev['opponent']}: score {ev['score']:.3f} "
                  f"({ev['wins']}W {ev['draws']}D {ev['losses']}L over {ev['games']} games)", flush=True)
            state = {"model": model.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                     "iter": it, "args": vars(a), "eval": ev}
            torch.save(state, out / "checkpoint.pt")
            if ev["score"] > best:
                best = ev["score"]; torch.save(state, out / "best.pt")
                print(f"  new best ({best:.3f}) -> {out/'best.pt'}", flush=True)

        if out_of_time:
            print(f"time budget of {a.hours} h reached at iteration {it}", flush=True)
            break

    env.close()
    json.dump({"args": vars(a), "best_eval_score": best, "history": history}, open(DATA / f"rl_{a.tag}.json", "w"), indent=1)
    print(f"done: best eval score {best:.3f}, {len(env.finished)} games played, {(time.time()-t0)/60:.1f} min", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DATA / "train_gpu" / "model_step11600.pt"))
    ap.add_argument("--tag", default="ppo"); ap.add_argument("--device", default="cpu"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=24)
    ap.add_argument("--opponent", choices=["random", "greedy", "stockfish"], default="greedy")
    ap.add_argument("--eval-opponent", choices=["random", "greedy", "stockfish"], default=None)
    ap.add_argument("--stockfish", default="stockfish"); ap.add_argument("--sf-depth", type=int, default=1); ap.add_argument("--sf-elo", type=int, default=1320)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--envs", type=int, default=32, help="games in parallel; also the width of every forward pass")
    ap.add_argument("--horizon", type=int, default=8, help="moves collected per game per iteration")
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--gamma", type=float, default=0.995); ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--shaping", type=float, default=0.5, help="weight on the potential-based material term")
    ap.add_argument("--temperature", type=float, default=1.0, help="softmax temperature the fly explores at")
    ap.add_argument("--ent-coef", type=float, default=0.01); ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--kl-ref-coef", type=float, default=0.05, help="pull towards the supervised policy; 0 disables the anchor")
    ap.add_argument("--lr-head", type=float, default=1e-4); ap.add_argument("--lr-gain", type=float, default=1e-4)
    ap.add_argument("--lr-bias", type=float, default=1e-7); ap.add_argument("--lr-critic", type=float, default=1e-3)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--train", choices=["heads", "gains", "all"], default="gains")
    ap.add_argument("--opening-plies", type=int, default=4); ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=10); ap.add_argument("--eval-games", type=int, default=64)
    ap.add_argument("--hours", type=float, default=0.0, help="wall-clock budget; stop cleanly when it runs out")
    ap.add_argument("--resume", default=None, help="continue from an rl checkpoint (model, critic and optimizer)")
    return ap.parse_args()


if __name__ == "__main__":
    main()
