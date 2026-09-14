"""Run the two check experiments at the chosen synaptic scale and export tick-by-tick activity
of every neuron that has a cell body in the Cloud Atlas, as compact sparse frames.

Frame format: for each tick, the top-K neurons by |rate| as (soma_index uint32, rate int8 * 127),
base64-encoded. Written to data/activity.json and injected into the viewer by build_viewer.py.
"""
import base64
import json
import sys
import numpy as np
import torch
from .graph import load, DATA
from .model import FlyBrain, graded_mask
from .checks import pick_sets

EXPERIMENTS = [
    ("sugar", "Sugar on the labellum", "sugar_grn", "Labellar sugar taste neurons are driven for the whole clip. Watch activity climb from the gnathal ganglion (green) into the feeding interneurons and the proboscis motor neurons."),
    ("light", "Light on the right eye", "R1-6_right", "The right eye's R1–R6 photoreceptors are driven. Photoreceptors inhibit the lamina, so the first wave is a dip (blue) that turns into excitation (amber) one and two synapses in: medulla, then T4/T5 and the lobula plate."),
]


def export(T=48, top_k=6000, device="cpu"):
    W, meta = load()
    cfg = json.load(open(DATA / "checks.json"))["chosen_config"]
    alpha = cfg["alpha"]
    brain = FlyBrain(W, device=device, graded_mask=graded_mask(meta), **cfg)
    sets = pick_sets(W, meta, np.random.default_rng(0))
    soma = meta.soma_index.to_numpy()
    has = soma >= 0
    out = {"config": cfg, "ticks": T, "dt_ms": 5, "experiments": []}
    for key, title, stim, blurb in EXPERIMENTS:
        _, frames = brain.run(brain.current(sets[stim]), T=T)
        stim_mask = np.zeros(len(meta), bool); stim_mask[sets[stim]] = True
        R = np.stack([f[:, 0].numpy() for f in frames])            # [T, N]
        R[:, ~has] = 0.0
        vmax = float(np.abs(R[:, ~stim_mask]).max()) or 1.0        # peak response outside the stimulated set
        fr_out = []
        for v in R:
            v = np.where(stim_mask, 0.0, v)                          # stimulated neurons are drawn separately
            order = np.argsort(-np.abs(v))[:top_k]
            order = order[np.abs(v[order]) > 0.02 * vmax]
            idx = soma[order].astype(np.uint32)
            val = np.clip(np.round(v[order] / vmax * 127), -127, 127).astype(np.int8)
            fr_out.append({"n": int(len(idx)), "idx": base64.b64encode(idx.tobytes()).decode(), "val": base64.b64encode(val.tobytes()).decode()})
        stim_idx = soma[sets[stim]]
        stim_idx = stim_idx[stim_idx >= 0].astype(np.uint32)
        active_peak = max(f["n"] for f in fr_out)
        out["experiments"].append({"key": key, "title": title, "blurb": blurb, "stimulated": int(len(sets[stim])), "peak_rate": vmax,
                                   "stim_idx": base64.b64encode(stim_idx.tobytes()).decode(), "frames": fr_out, "peak_active": active_peak})
        print(key, "frames", len(fr_out), "peak active (with soma, |r|>0.02):", active_peak)
    json.dump(out, open(DATA / "activity.json", "w"))
    print("wrote data/activity.json", round((DATA / "activity.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    export(device=sys.argv[1] if len(sys.argv) > 1 else "cpu")
