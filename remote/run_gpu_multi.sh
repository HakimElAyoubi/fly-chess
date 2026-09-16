#!/bin/bash
# Multi-GPU schedule on a rented box. Usage: bash remote/run_gpu_multi.sh <hours_budget> [warm_games] [full_games] [batch_per_gpu]
#  1. tables, graph, checks; stream a warm-up set and, in the background, the full set
#  2. DDP smoke test (3 steps)
#  3. stage 1: train on the warm-up set for 3,000 steps
#  4. stage 2: resume on the full set until the time budget runs out (checkpoint every 200 steps)
set -e
cd "$(dirname "$0")/.."
HOURS=${1:-4.5}; WARM=${2:-150000}; FULL=${3:-600000}; BATCH=${4:-128}
pip install -q numpy pandas pyarrow scipy python-chess zstandard requests 2>&1 | grep -v -i "warning\|notice" || true
mkdir -p data
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
for f in body-annotations-male-cns-v1.0-minconf-0.5.feather body-neurotransmitters-male-cns-v1.0.feather connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather; do
  [ -f data/$f ] || curl -sS -o data/$f $B/$f
done
# streams first (they are the slow part), in the background
[ -f data/positions_full.jsonl.gz ] || (nohup python -m flybrain.data $FULL full > data/stream_full.log 2>&1 &)
[ -f data/positions_warm.jsonl.gz ] || python -m flybrain.data $WARM warm > data/stream_warm.log 2>&1
[ -f data/graph_brain.npz ] || python -m flybrain.graph
[ -f data/checks.json ] || python -m flybrain.checks cuda
NG=$(nvidia-smi -L | wc -l)
echo "=== DDP smoke test on $NG GPUs"
torchrun --nproc_per_node=$NG -m flybrain.train --ddp --edge-gains --positions data/positions_smoke.jsonl.gz --steps 3 --batch 8 --n-eval 16 --eval-every 3 --tag smoke
echo "=== stage 1: warm-up set"
START=$(date +%s)
torchrun --nproc_per_node=$NG -m flybrain.train --ddp --edge-gains --positions data/positions_warm.jsonl.gz --steps 3000 --batch $BATCH --tag gpu --eval-every 200 --n-eval 512
echo "=== waiting for the full set"
while [ ! -f data/positions_full.jsonl.gz ] || pgrep -f "flybrain.data $FULL full" > /dev/null; do sleep 30; done
tail -1 data/stream_full.log
LEFT=$(python -c "import time; print(max(600, int($HOURS*3600 - (time.time() - $START))))")
echo "=== stage 2: full set, $LEFT s left in the budget"
timeout ${LEFT}s torchrun --nproc_per_node=$NG -m flybrain.train --ddp --edge-gains --positions data/positions_full.jsonl.gz --resume data/train_gpu/checkpoint.pt --steps 100000 --batch $BATCH --tag gpu --eval-every 200 --n-eval 512 || echo "=== stage 2 ended (timeout or error, exit $?)"
echo "=== done"
