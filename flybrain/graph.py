"""Build the digital fly's wiring from MaleCNS v1.0.

Neurons: every traced neuron on the brain side of the neck (optic lobes, central brain,
descending/ascending neurons, brain sensory axons). Edges: every neuron-to-neuron connection
in the traced-only weights table between those neurons. Sign: from the presynaptic neuron's
predicted neurotransmitter (Dale's law: one sign per neuron).

Outputs (in data/):
  graph_brain.npz         CSR matrix W[post, pre] = synapse count * sign(pre), float32
  graph_brain_meta.feather one row per neuron: bodyId, type, superclass, side, sign, soma_index ...
  graph_brain_summary.json counts for the report
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.feather as pf
import scipy.sparse as sp

DATA = Path(__file__).resolve().parent.parent / "data"
ANN = DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = DATA / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"
NT = DATA / "body-neurotransmitters-male-cns-v1.0.feather"

BRAIN_SUPERCLASSES = {
    "ol_intrinsic", "cb_intrinsic", "visual_projection", "visual_centrifugal", "visual_projection_tbc",
    "ol_sensory", "cb_sensory", "cb_sensory_tbc",
    "descending_neuron", "ascending_neuron", "sensory_ascending", "sensory_ascending_tbc", "sensory_descending",
    "cb_motor", "cb_endocrine", "cb_efferent", "efferent_ascending", "efferent_descending",
}
# fast transmitters carry a sign; the three modulators (dopamine, octopamine, serotonin) are
# silent in version one, exactly as the plan says. Histamine is the photoreceptor transmitter
# and is inhibitory (chloride channels on lamina cells).
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0,
        "dopamine": 0.0, "octopamine": 0.0, "serotonin": 0.0}


def build(verbose=True):
    ann = pd.read_feather(ANN)
    traced = ann[ann.status == "Traced"].reset_index(drop=True)
    # the Cloud Atlas soma layer is exactly this subset in this order (see build_cloud.py)
    soma_rows = traced[traced.somaLocation.notna()].reset_index(drop=True)
    soma_index = pd.Series(np.arange(len(soma_rows)), index=soma_rows.bodyId.values)

    neurons = traced[traced.superclass.isin(BRAIN_SUPERCLASSES)].reset_index(drop=True)
    N = len(neurons)
    idx = pd.Series(np.arange(N), index=neurons.bodyId.values)

    nt = pf.read_table(NT, columns=["body", "consensus_nt", "celltype_predicted_nt"]).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")
    cons = nt.consensus_nt.reindex(neurons.bodyId.values).to_numpy(dtype=object)
    ctyp = nt.celltype_predicted_nt.reindex(neurons.bodyId.values).to_numpy(dtype=object)
    nt_used = np.array([c if c in SIGN else (t if t in SIGN else "assumed_acetylcholine") for c, t in zip(cons, ctyp)], dtype=object)
    sign = np.array([SIGN.get(x, 1.0) for x in nt_used], dtype=np.float32)

    w = pf.read_table(WEIGHTS, columns=["body_pre", "body_post", "weight"]).to_pandas()
    keep = w.body_pre.isin(idx.index) & w.body_post.isin(idx.index)
    w = w[keep]
    pre = idx[w.body_pre.values].to_numpy()
    post = idx[w.body_post.values].to_numpy()
    vals = (w.weight.to_numpy().astype(np.float32)) * sign[pre]
    W = sp.csr_matrix((vals, (post, pre)), shape=(N, N), dtype=np.float32)
    W.sum_duplicates()
    W.eliminate_zeros()  # modulatory (sign 0) edges vanish here

    # side: soma side where known; photoreceptors have no soma in the volume, so take the
    # synapse-weighted majority side of their postsynaptic partners.
    side = neurons.somaSide.fillna("").to_numpy(dtype=object)
    is_pr = (neurons.superclass == "ol_sensory").to_numpy()
    pr_edges = is_pr[pre]
    post_side = side[post[pr_edges]]
    df = pd.DataFrame({"pre": pre[pr_edges], "side": post_side, "w": np.abs(vals[pr_edges])})
    df = df[df.side.isin(["L", "R"])]
    tally = df.pivot_table(index="pre", columns="side", values="w", aggfunc="sum", fill_value=0.0)
    for s in ("L", "R"):
        if s not in tally: tally[s] = 0.0
    pr_side = np.where(tally["R"] > tally["L"], "R", "L")
    side[tally.index.to_numpy()] = pr_side
    side[(side == "") & is_pr] = "unknown"

    meta = pd.DataFrame({
        "idx": np.arange(N), "bodyId": neurons.bodyId.values,
        "type": neurons.type.values, "superclass": neurons.superclass.values,
        "class": neurons["class"].values, "subclass": neurons.subclass.values,
        "side": side, "nt": nt_used, "sign": sign,
        "synonyms": neurons.synonyms.map(lambda v: v if isinstance(v, str) else None).values,
        "soma_index": soma_index.reindex(neurons.bodyId.values).fillna(-1).astype(int).values,
        "in_synapses": np.asarray(np.abs(W).sum(axis=1)).ravel(),
        "out_synapses": np.asarray(np.abs(W).sum(axis=0)).ravel(),
    })
    sp.save_npz(DATA / "graph_brain.npz", W)
    meta.to_feather(DATA / "graph_brain_meta.feather")
    summary = {
        "neurons": int(N), "edges": int(W.nnz), "synapses": float(np.abs(W).sum()),
        "edges_before_sign_filter": int(len(w)),
        "sign_counts": {k: int((meta.nt == k).sum()) for k in sorted(set(nt_used))},
        "photoreceptors_by_side": meta[meta.superclass == "ol_sensory"].side.value_counts().to_dict(),
        "superclass_counts": meta.superclass.value_counts().to_dict(),
        "neurons_with_soma_in_atlas": int((meta.soma_index >= 0).sum()),
    }
    json.dump(summary, open(DATA / "graph_brain_summary.json", "w"), indent=1)
    if verbose:
        print(json.dumps(summary, indent=1))
    return W, meta


def load():
    W = sp.load_npz(DATA / "graph_brain.npz").tocsr()
    meta = pd.read_feather(DATA / "graph_brain_meta.feather")
    return W, meta


if __name__ == "__main__":
    build()
