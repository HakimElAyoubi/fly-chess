"""Rate model of the digital fly.

r <- r + k * ( f(alpha * W r + I + b) - r ),   k = dt / tau

W is the signed connectome (post x pre, synapse counts with the presynaptic sign), alpha is a
global synaptic scale, I is injected current (stimulus), b a per-neuron bias. f is tanh by
default: activity is measured relative to rest, so a neuron can be pushed above or below its
baseline. That is what lets inhibitory pathways in the graded optic lobe (photoreceptor ->
lamina -> medulla) carry signal by disinhibition. 'relu' gives a silent-at-rest spiking-style
network with threshold 1.
"""
import time
import numpy as np
import scipy.sparse as sp
import torch


class FlyBrain:
    def __init__(self, W: sp.csr_matrix, alpha=0.05, dt_over_tau=0.25, nonlin="tanh", device="cpu", theta=1.0, normalize=False,
                 graded_mask=None, slope=1.0, alpha_spk=0.05):
        """nonlin 'mixed': neurons flagged in graded_mask (the optic lobe's non-spiking columnar
        cells and photoreceptors) are tanh units whose inputs are normalised to sum to 1 and
        scaled by `alpha` (a gain; the resting state is stable for alpha <= 1). Every other neuron
        is a spiking-style threshold unit fed by raw synapse counts times `alpha_spk`: it is
        silent at rest, crosses threshold when 1/alpha_spk synapses from fully active partners
        are on, and saturates at twice that: rate = clamp(slope * (x - theta), 0, 1), theta = 1."""
        W = W.tocsr().astype(np.float32)
        tot = np.asarray(np.abs(W).sum(axis=1)).ravel()
        inv = np.where(tot > 0, 1.0 / np.maximum(tot, 1e-9), 0.0).astype(np.float32)
        if nonlin == "mixed":
            gm = np.asarray(graded_mask, bool)
            row_scale = np.where(gm, inv * alpha, alpha_spk).astype(np.float32)
            W = (sp.diags(row_scale) @ W).tocsr().astype(np.float32)
            alpha = 1.0                       # already folded into the rows
        elif normalize:
            # each neuron's inputs sum to 1 in magnitude, so alpha is the gain a neuron sees when
            # every one of its inputs is fully active. Row-abs-sums of 1 bound the spectral radius
            # by 1, so the resting state is stable for alpha <= 1 (Gershgorin).
            W = (sp.diags(inv) @ W).tocsr().astype(np.float32)
        self.N = W.shape[0]
        self.alpha, self.k, self.nonlin, self.theta, self.normalize, self.slope, self.alpha_spk = alpha, dt_over_tau, nonlin, theta, normalize, slope, alpha_spk
        self.device = torch.device(device)
        self.graded = None if graded_mask is None else torch.as_tensor(np.asarray(graded_mask, bool)).view(-1, 1).to(self.device)
        crow = torch.from_numpy(W.indptr.astype(np.int64))
        col = torch.from_numpy(W.indices.astype(np.int64))
        val = torch.from_numpy(W.data) * alpha
        self.W = torch.sparse_csr_tensor(crow, col, val, size=W.shape).to(self.device)
        self.bias = torch.zeros(self.N, 1, device=self.device)

    def f(self, x):
        if self.nonlin == "tanh":
            return torch.tanh(x)
        if self.nonlin == "relu_sat":          # silent below threshold, saturates at 1: LIF-like
            return torch.clamp(self.slope * (x - self.theta), min=0.0, max=1.0)
        if self.nonlin == "mixed":
            return torch.where(self.graded, torch.tanh(x), torch.clamp(self.slope * (x - self.theta), min=0.0, max=1.0))
        return torch.relu(x - self.theta)

    @torch.no_grad()
    def run(self, I, T=48, record_every=1, r0=None):
        """I: [N, B] currents (constant over time) or callable t -> [N, B]. Returns final r [N,B]
        and a list of recorded r tensors (on CPU)."""
        B = I(0).shape[1] if callable(I) else I.shape[1]
        r = torch.zeros(self.N, B, device=self.device) if r0 is None else r0.clone()
        frames = []
        for t in range(T):
            It = I(t) if callable(I) else I
            x = torch.sparse.mm(self.W, r) + It + self.bias
            r = r + self.k * (self.f(x) - r)
            if record_every and (t % record_every == record_every - 1):
                frames.append(r.detach().cpu().clone())
        return r, frames

    def current(self, indices, amplitude=2.0, B=1):
        I = torch.zeros(self.N, B, device=self.device)
        I[torch.as_tensor(np.asarray(indices), device=self.device)] = amplitude
        return I

    def time_forward(self, B=128, T=32, reps=2):
        I = torch.zeros(self.N, B, device=self.device)
        self.run(I, T=4, record_every=0)  # warm-up
        if self.device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            self.run(I, T=T, record_every=0)
        if self.device.type == "mps":
            torch.mps.synchronize()
        return (time.perf_counter() - t0) / reps


def graded_mask(meta):
    """Optic-lobe neurons (columnar cells and photoreceptors) are graded, non-spiking cells."""
    return meta.superclass.isin(["ol_intrinsic", "ol_sensory"]).to_numpy()
