"""The digital fly as a chess policy (Phase 3).

Board -> right eye (Eye.encode) -> the connectome-constrained network run for T ticks ->
descending-neuron rates -> a linear move readout (from-square x to-square = 4,096 logits, plus
5 promotion logits) and a linear value head (win / draw / loss).

What is learned, in this Mac-sized variant: a positive output gain per neuron, a positive input
gain per neuron, and a bias per neuron (3 x 144,209 parameters), plus the two linear heads.
The wiring, the signs and the synapse counts stay fixed. Per-edge gains (21 M parameters) are
the GPU variant and use the same code path with `edge_gains=True`.
"""
import json
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import chess
from .graph import load, DATA
from .model import FlyBrain

N_MOVES = 64 * 64
PROMO = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
PROMO_INV = {v: k for k, v in PROMO.items()}


def move_index(move: chess.Move):
    return move.from_square * 64 + move.to_square, PROMO[move.promotion]


def index_move(idx, promo, board=None):
    m = chess.Move(idx // 64, idx % 64, promotion=PROMO_INV[int(promo)] if promo else None)
    if board is not None and m.promotion is None:
        pc = board.piece_at(m.from_square)
        if pc is not None and pc.piece_type == chess.PAWN and chess.square_rank(m.to_square) in (0, 7):
            m = chess.Move(m.from_square, m.to_square, promotion=chess.QUEEN)
    return m


class SpMM(torch.autograd.Function):
    """y = W @ r for a fixed sparse W; the backward pass uses the precomputed transpose."""
    @staticmethod
    def forward(ctx, W, WT, r):
        ctx.WT = WT
        return torch.sparse.mm(W, r)

    @staticmethod
    def backward(ctx, g):
        return None, None, torch.sparse.mm(ctx.WT, g)


class EdgeSpMM(torch.autograd.Function):
    """y = W(v) @ r with learnable values v (per-edge gains).
    backward: dL/dr = W(v)^T g, using the precomputed transposed pattern and a permutation of v;
              dL/dv_e = sum_b g[row_e, b] r[col_e, b], a sampled dense-dense product on W's pattern."""
    @staticmethod
    def forward(ctx, v, r, crow, col, crowT, colT, perm, pattern, N):
        W = torch.sparse_csr_tensor(crow, col, v, size=(N, N))
        ctx.save_for_backward(v, r, crow, col, crowT, colT, perm)
        ctx.pattern, ctx.N = pattern, N
        return torch.sparse.mm(W, r)

    @staticmethod
    def backward(ctx, g):
        v, r, crow, col, crowT, colT, perm = ctx.saved_tensors
        WT = torch.sparse_csr_tensor(crowT, colT, v[perm], size=(ctx.N, ctx.N))
        grad_r = torch.sparse.mm(WT, g)
        g = g.contiguous(); rT = r.T.contiguous()
        try:
            grad_v = torch.sparse.sampled_addmm(ctx.pattern, g, rT).values()
        except (RuntimeError, NotImplementedError):                          # no SDDMM on this device: chunk over edges
            rows = torch.repeat_interleave(torch.arange(ctx.N, device=g.device), torch.diff(crow))
            grad_v = torch.empty_like(v)
            step = 2_000_000
            for s in range(0, len(v), step):
                grad_v[s:s + step] = (g[rows[s:s + step]] * r[col[s:s + step]]).sum(1)
        return grad_v, grad_r, None, None, None, None, None, None, None


class FlyPolicy(nn.Module):
    def __init__(self, device="cpu", ticks=24, edge_gains=False):
        super().__init__()
        W, meta = load()
        cfg = json.load(open(DATA / "checks.json"))["chosen_config"]
        self.meta, self.cfg, self.ticks, self.k = meta, cfg, ticks, 0.25
        base = FlyBrain(W, device="cpu", **cfg)                       # builds the normalised, gained matrix
        Wn = sp.csr_matrix((base.W.values().numpy(), base.W.col_indices().numpy(), base.W.crow_indices().numpy()), shape=(base.N, base.N))
        self.N = base.N
        self.device = torch.device(device)
        crow = torch.from_numpy(Wn.indptr.astype(np.int64)); col = torch.from_numpy(Wn.indices.astype(np.int64)); val = torch.from_numpy(Wn.data.astype(np.float32))
        self.edge_gains = edge_gains
        if edge_gains:
            self.register_buffer("crow", crow.to(self.device), persistent=False); self.register_buffer("col", col.to(self.device), persistent=False)
            self.register_buffer("base_val", val.to(self.device), persistent=False)
            # transposed pattern and the permutation that maps its entries back to W's entries
            tag = sp.csr_matrix((np.arange(Wn.nnz, dtype=np.float64), Wn.indices, Wn.indptr), shape=Wn.shape).T.tocsr()
            self.register_buffer("crowT", torch.from_numpy(tag.indptr.astype(np.int64)).to(self.device), persistent=False)
            self.register_buffer("colT", torch.from_numpy(tag.indices.astype(np.int64)).to(self.device), persistent=False)
            self.register_buffer("perm", torch.from_numpy(tag.data.astype(np.int64)).to(self.device), persistent=False)
            self.pattern = torch.sparse_csr_tensor(self.crow, self.col, torch.zeros(len(val), device=self.device), size=(self.N, self.N))
            self.log_edge_gain = nn.Parameter(torch.zeros(len(val), device=self.device))
        else:
            self.W = torch.sparse_csr_tensor(crow, col, val, size=(self.N, self.N)).to(self.device)
            WT = Wn.T.tocsr()
            self.WT = torch.sparse_csr_tensor(torch.from_numpy(WT.indptr.astype(np.int64)), torch.from_numpy(WT.indices.astype(np.int64)),
                                              torch.from_numpy(WT.data.astype(np.float32)), size=(self.N, self.N)).to(self.device)
        self.log_gain_out = nn.Parameter(torch.zeros(self.N, 1, device=self.device))   # output gain per neuron
        self.log_gain_in = nn.Parameter(torch.zeros(self.N, 1, device=self.device))    # input gain per neuron
        self.bias = nn.Parameter(torch.zeros(self.N, 1, device=self.device))
        self.register_buffer("dn", torch.as_tensor(meta.index[meta.superclass == "descending_neuron"].to_numpy().copy(), device=self.device), persistent=False)
        # fixed standardisation of the descending-neuron rates before the linear heads (set by
        # calibrate(); the untrained rates are ~1e-5, far too small for the heads to learn from)
        self.register_buffer("dn_mean", torch.zeros(len(self.dn), device=self.device))
        self.register_buffer("dn_scale", torch.ones(len(self.dn), device=self.device))
        self.readout = nn.Linear(len(self.dn), N_MOVES + 5).to(self.device)
        self.value = nn.Linear(len(self.dn), 3).to(self.device)

    @torch.no_grad()
    def calibrate(self, I):
        """Set the descending-neuron standardisation from a batch of input currents [N, B]."""
        dn = self.run(I)
        self.dn_mean.copy_(dn.mean(0))
        scale = 1.0 / (dn.std(0) + 1e-12)
        self.dn_scale.copy_(torch.minimum(scale, 10.0 * scale.median()))          # cap: a silent neuron must not become a noise amplifier

    def run_full(self, I):
        """I: [N, B] input currents -> every neuron's rate [N, B] at the final tick."""
        r = torch.zeros(self.N, I.shape[1], device=self.device)
        g_out, g_in = torch.exp(self.log_gain_out), torch.exp(self.log_gain_in)
        for _ in range(self.ticks):
            pre = g_out * r
            if self.edge_gains:
                syn = EdgeSpMM.apply(self.base_val * torch.exp(self.log_edge_gain), pre, self.crow, self.col, self.crowT, self.colT, self.perm, self.pattern, self.N)
            else:
                syn = SpMM.apply(self.W, self.WT, pre)
            r = r + self.k * (torch.tanh(g_in * syn + I + self.bias) - r)
        return r

    def run(self, I):
        """I: [N, B] input currents -> descending-neuron rates [B, n_dn]."""
        r = torch.zeros(self.N, I.shape[1], device=self.device)
        g_out, g_in = torch.exp(self.log_gain_out), torch.exp(self.log_gain_in)
        for _ in range(self.ticks):
            pre = g_out * r
            if self.edge_gains:
                syn = EdgeSpMM.apply(self.base_val * torch.exp(self.log_edge_gain), pre, self.crow, self.col, self.crowT, self.colT, self.perm, self.pattern, self.N)
            else:
                syn = SpMM.apply(self.W, self.WT, pre)
            x = g_in * syn + I + self.bias
            r = r + self.k * (torch.tanh(x) - r)
        return r[self.dn].T

    def forward(self, I):
        dn = (self.run(I) - self.dn_mean) * self.dn_scale
        out = self.readout(dn)
        return out[:, :N_MOVES], out[:, N_MOVES:], self.value(dn), dn

    def n_params(self):
        return {n: int(p.numel()) for n, p in self.named_parameters()}


def load_into(model, state, where="", strict_report=True):
    """Load weights and report what did not match, so a mismatched variant cannot pass silently."""
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("dn")]
    if strict_report and (missing or result.unexpected_keys):
        print(f"  weights {where}: {len(state)} tensors loaded"
              + (f" | MISSING {missing}" if missing else "")
              + (f" | IGNORED {list(result.unexpected_keys)}" if result.unexpected_keys else ""), flush=True)
    if "log_edge_gain" in state and not model.edge_gains:
        raise ValueError("these weights have per-edge gains but the model was built without them; "
                         "build FlyPolicy(edge_gains=True)")
    return result


def variant_of(path, device="cpu"):
    """True when a checkpoint holds per-edge gains."""
    ck = torch.load(path, map_location=device, weights_only=False)
    return "log_edge_gain" in ck["model"], ck


def legal_mask(boards, device="cpu"):
    m = torch.full((len(boards), N_MOVES), float("-inf"), device=device)
    for i, b in enumerate(boards):
        for mv in b.legal_moves:
            m[i, mv.from_square * 64 + mv.to_square] = 0.0
    return m
