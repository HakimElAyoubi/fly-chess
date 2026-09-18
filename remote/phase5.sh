#!/bin/bash
# Phase 5 on the box: the ablation sweep. Usage: bash remote/phase5.sh <n_positions>
set -e
cd "$(dirname "$0")/.."
N=${1:-2048}
pip install -q numpy pandas pyarrow scipy python-chess zstandard requests 2>&1 | grep -v -i "warning\|notice" || true
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
for f in body-annotations-male-cns-v1.0-minconf-0.5.feather body-neurotransmitters-male-cns-v1.0.feather connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather; do
  [ -f data/$f ] || curl -sS -o data/$f $B/$f
done
[ -f data/graph_brain.npz ] || python -m flybrain.graph
[ -f data/checks.json ] || python -m flybrain.checks cuda
sha256sum -c data/weights.sha256
echo "=== ablation sweep, $N positions per group"
python -m flybrain.ablate --device cuda --n $N --batch 256 --min-neurons 50
echo "=== done"
