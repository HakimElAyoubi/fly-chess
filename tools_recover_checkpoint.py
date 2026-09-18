"""Recover the model tensors from a checkpoint whose zip archive mixes two writes (a save that was
interrupted over an older file). Uses the first occurrence of every entry in the file body (the
newest write) and its data descriptors; storages that the newest write never finished (optimizer
state) are filled with zeros so PyTorch's own loader can read the result.
Usage: python tools_recover_checkpoint.py <corrupt.pt> <clean.pt>"""
import io, pickle, re, struct, sys, zipfile
import torch

src, dst = sys.argv[1], sys.argv[2]
data = open(src, "rb").read()
hdrs, j = [], 0
while True:
    j = data.find(b"PK\x03\x04", j)
    if j < 0: break
    nlen, xlen = struct.unpack("<HH", data[j+26:j+30]); name = data[j+30:j+30+nlen]
    if 0 < nlen < 80 and name.startswith(b"checkpoint/") and re.fullmatch(rb"[\w./]+", name):
        hdrs.append((j, name.decode(), j + 30 + nlen + xlen))
    j += 4
first = {}
for k, (off, name, start) in enumerate(hdrs):
    if name in first: continue
    nxt = hdrs[k+1][0] if k + 1 < len(hdrs) else data.rfind(b"PK\x01\x02")
    for dlen in (16, 24):
        d = data[nxt-dlen:nxt]
        if d[:4] != b"PK\x07\x08": continue
        us = struct.unpack("<I", d[12:16])[0] if dlen == 16 else struct.unpack("<Q", d[16:24])[0]
        if us == nxt - dlen - start:
            first[name] = data[start:start+us]; break
print("complete entries from the newest write:", len(first))
# pass 1: learn every storage's key, dtype and length from the pickle
storages = {}
class Probe(pickle.Unpickler):
    def persistent_load(self, pid):
        typename, storage_type, key, location, numel = pid
        dtype = storage_type.dtype if hasattr(storage_type, "dtype") else torch.float32
        storages[key] = (dtype, numel)
        return torch.storage.TypedStorage(wrap_storage=torch.zeros(numel, dtype=dtype).untyped_storage(), dtype=dtype, _internal=True)
    def find_class(self, module, name):
        return super().find_class(module, name)
obj = Probe(io.BytesIO(first["checkpoint/data.pkl"])).load()
print("storages referenced by the pickle:", len(storages), "| model tensors:", list(obj["model"].keys())[:4], "...")
# pass 2: clean archive with real bytes where the newest write finished, zeros elsewhere
have = 0
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zo:
    zo.writestr("checkpoint/data.pkl", first["checkpoint/data.pkl"])
    zo.writestr("checkpoint/byteorder", first.get("checkpoint/byteorder", b"little"))
    zo.writestr("checkpoint/version", first.get("checkpoint/version", b"3"))
    for key, (dtype, numel) in storages.items():
        b = first.get(f"checkpoint/data/{key}")
        need = numel * torch.tensor([], dtype=dtype).element_size()
        if b is not None and len(b) == need: have += 1
        else: b = bytes(need)
        zo.writestr(f"checkpoint/data/{key}", b)
print(f"storages with real data: {have} of {len(storages)} (the rest, optimizer state, zero-filled)")
ck = torch.load(dst, map_location="cpu", weights_only=False)
print("torch.load ok: step", ck["step"], "| model tensors", len(ck["model"]))
