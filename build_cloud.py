"""Sample the MaleCNS neuropil ROI volumes and neuron soma positions into a
compact binary point cloud, then embed it into viewer.html.

Output units: micrometers * 10 stored as int16 (0.1 um resolution).
"""
import json, base64, struct, sys
import numpy as np
import pandas as pd
from cloudvolume import CloudVolume

BUCKET = "precomputed://gs://flyem-male-cns/"
rng = np.random.default_rng(7)

def load_roi(path, mip_res_nm):
    vol = CloudVolume(BUCKET + path, use_https=True, mip=[mip_res_nm]*3, fill_missing=True, progress=False)
    print(path, "shape", vol.shape, "res", vol.resolution, flush=True)
    data = np.asarray(vol[:, :, :]).squeeze()
    return data, vol.resolution

def roi_names(path):
    import requests
    url = f"https://storage.googleapis.com/flyem-male-cns/{path}/segment_properties/info"
    d = requests.get(url, timeout=60).json()
    ids = d["inline"]["ids"]
    for pr in d["inline"]["properties"]:
        if pr["type"] == "label":
            return {int(i): v for i, v in zip(ids, pr["values"])}
    raise RuntimeError("no label property")

def sample_points(data, res_nm, id_offset, names, target_total, min_per_region=400):
    """Return (xyz_um float32 [N,3], label uint16 [N]) sampled proportional to volume."""
    ids, counts = np.unique(data, return_counts=True)
    keep = ids != 0
    ids, counts = ids[keep], counts[keep]
    total = counts.sum()
    out_xyz, out_lab, meta = [], [], []
    for rid, cnt in zip(ids, counts):
        if int(rid) not in names:
            continue
        n = max(min_per_region, int(target_total * cnt / total))
        n = min(n, cnt)
        idx = np.flatnonzero(data.ravel() == rid)
        pick = rng.choice(idx, size=n, replace=False)
        x, y, z = np.unravel_index(pick, data.shape)
        # jitter within voxel so the cloud doesn't show the grid
        xyz = np.stack([x, y, z], 1).astype(np.float32) + rng.random((n, 3), dtype=np.float32)
        xyz *= np.asarray(res_nm, dtype=np.float32) / 1000.0  # -> um
        out_xyz.append(xyz)
        out_lab.append(np.full(n, id_offset + int(rid), np.uint16))
        vol_um3 = float(cnt) * np.prod(np.asarray(res_nm) / 1000.0)
        meta.append({"id": id_offset + int(rid), "name": names[int(rid)], "points": int(n),
                     "volume_um3": round(vol_um3), "centroid": [round(float(v), 1) for v in xyz.mean(0)]})
        print(f"  {names[int(rid)]:22s} voxels={cnt:9d} sampled={n}", flush=True)
    return np.concatenate(out_xyz), np.concatenate(out_lab), meta

def main():
    brain, bres = load_roi("rois/fullbrain-roi-v5", 1024)
    bnames = roi_names("rois/fullbrain-roi-v5")
    bxyz, blab, bmeta = sample_points(brain, bres, 0, bnames, target_total=420_000)
    del brain

    vnc, vres = load_roi("rois/malecns-vnc-neuropil-roi-v0", 2048)
    vnames = roi_names("rois/malecns-vnc-neuropil-roi-v0")
    vxyz, vlab, vmeta = sample_points(vnc, vres, 1000, vnames, target_total=220_000)
    del vnc

    xyz = np.concatenate([bxyz, vxyz]); lab = np.concatenate([blab, vlab])

    # neuron cell bodies (soma / nucleus) from the v1.0 annotation table, 8 nm voxel units
    ann = pd.read_feather("data/body-annotations-male-cns-v1.0-minconf-0.5.feather")
    ann = ann[(ann.status == "Traced") & ann.somaLocation.notna()]
    sxyz = np.stack(ann.somaLocation.to_numpy()).astype(np.float32) * 8.0 / 1000.0
    sclasses = sorted(ann.superclass.fillna("unknown").unique())
    sidx = {c: i for i, c in enumerate(sclasses)}
    slab = ann.superclass.fillna("unknown").map(sidx).to_numpy().astype(np.uint16)
    print("somas", len(sxyz), "superclasses", len(sclasses))

    def pack(p, l):
        q = np.clip(np.round(p * 10.0), -32768, 32767).astype(np.int16)  # 0.1 um units
        return base64.b64encode(q.tobytes()).decode(), base64.b64encode(l.tobytes()).decode()

    roi_pos, roi_lab = pack(xyz, lab)
    soma_pos, soma_lab = pack(sxyz, slab)
    payload = {
        "roi": {"n": int(len(xyz)), "pos": roi_pos, "lab": roi_lab, "regions": bmeta + vmeta},
        "soma": {"n": int(len(sxyz)), "pos": soma_pos, "lab": soma_lab, "classes": sclasses,
                 "class_counts": {c: int((slab == i).sum()) for c, i in sidx.items()}},
        "bounds_um": [[round(float(v), 1) for v in xyz.min(0)], [round(float(v), 1) for v in xyz.max(0)]],
    }
    with open("data/cloud.json", "w") as f:
        json.dump(payload, f)
    print("wrote data/cloud.json", sum(len(v) for v in (roi_pos, roi_lab, soma_pos, soma_lab)) / 1e6, "MB base64")

if __name__ == "__main__":
    main()
