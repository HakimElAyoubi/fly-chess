#!/bin/bash
# Fine-tune on Stockfish labels. Runs on the box. Usage: bash remote/finetune.sh <steps> <batch>
set -e
cd "$(dirname "$0")/.."
STEPS=${1:-4000}; BATCH=${2:-128}; POS=${3:-data/positions_all_sf.jsonl.gz}
pip install -q numpy pandas pyarrow scipy python-chess zstandard requests 2>&1 | grep -v -i "warning\|notice" || true
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
for f in body-annotations-male-cns-v1.0-minconf-0.5.feather body-neurotransmitters-male-cns-v1.0.feather connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather; do
  [ -f data/$f ] || curl -sS -o data/$f $B/$f
done
[ -f data/graph_brain.npz ] || python -m flybrain.graph
[ -f data/checks.json ] || python -m flybrain.checks cuda
sha256sum -c data/weights.sha256 || { echo "WEIGHTS CORRUPT IN TRANSFER"; exit 1; }
NG=$(nvidia-smi -L | wc -l)
D="--positions $POS --init-weights data/train_gpu/model_step11600.pt --n-eval 512"
echo "dataset: $POS ($(gzip -cd $POS | wc -l) positions)"
echo "=== baseline: the loaded weights, before any fine-tuning"
python -m flybrain.train --device cuda $D --label-field move    --eval-only --batch 64 --tag ft
python -m flybrain.train --device cuda $D --label-field sf_move --eval-only --batch 64 --tag ft
echo "=== fine-tuning on Stockfish's choice, value from its evaluation ($NG GPUs)"
if [ "$NG" -gt 1 ]; then
  torchrun --nproc_per_node=$NG -m flybrain.train --ddp $D --label-field sf_move --value-from eval \
    --steps $STEPS --batch $BATCH --eval-every 200 --tag ft
else
  python -m flybrain.train --device cuda $D --label-field sf_move --value-from eval \
    --steps $STEPS --batch $BATCH --eval-every 200 --tag ft
fi
echo "=== after: agreement with the human move, for comparison"
python -m flybrain.train --device cuda --positions $POS \
  --init-weights data/train_ft/checkpoint.pt --label-field move --eval-only --batch 64 --tag ft
echo "=== done"
