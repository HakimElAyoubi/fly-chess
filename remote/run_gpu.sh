#!/bin/bash
# Runs on the rented GPU box. Usage: bash remote/run_gpu.sh <n_games> <steps> <batch> <tag> [extra train args]
set -e
cd "$(dirname "$0")/.."
N_GAMES=${1:-40000}; STEPS=${2:-4000}; BATCH=${3:-128}; TAG=${4:-gpu}; shift 4 || true
pip install -q numpy pandas pyarrow scipy python-chess zstandard requests 2>&1 | grep -v -i "warning\|notice" || true
mkdir -p data
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
for f in body-annotations-male-cns-v1.0-minconf-0.5.feather body-neurotransmitters-male-cns-v1.0.feather connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather; do
  [ -f data/$f ] || curl -sS -o data/$f $B/$f
done
[ -f data/graph_brain.npz ] || python -m flybrain.graph
[ -f data/checks.json ] || python -m flybrain.checks cuda
[ -f data/positions_$TAG.jsonl.gz ] || python -m flybrain.data $N_GAMES $TAG
NG=$(nvidia-smi -L | wc -l)
if [ "$NG" -gt 1 ]; then
  torchrun --nproc_per_node=$NG -m flybrain.train --ddp --edge-gains --positions data/positions_$TAG.jsonl.gz --steps $STEPS --batch $BATCH --tag $TAG --eval-every 200 --n-eval 512 "$@"
else
  python -m flybrain.train --device cuda --edge-gains --positions data/positions_$TAG.jsonl.gz --steps $STEPS --batch $BATCH --tag $TAG --eval-every 200 --n-eval 512 "$@"
fi
