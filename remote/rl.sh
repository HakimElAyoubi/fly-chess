#!/bin/bash
# PPO on a rented box. Usage: bash remote/rl.sh <hours> [opponent] [envs] [horizon] [tag]
# Expects the graph, the checks config and the supervised weights to have been pushed already.
set -e
cd "$(dirname "$0")/.."
HOURS=${1:-2.5}; OPP=${2:-greedy}; ENVS=${3:-96}; HOR=${4:-16}; TAG=${5:-ppo}
pip install -q numpy pandas pyarrow scipy python-chess 2>&1 | grep -vi "warning\|notice" || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
(setsid nohup bash remote/selfstop_watch.sh rl.log 20 > /dev/null 2>&1 &)
python -m flybrain.rl --device cuda --tag "$TAG" \
  --weights data/train_gpu/model_step11600.pt \
  --opponent "$OPP" --eval-opponent "$OPP" \
  --envs "$ENVS" --horizon "$HOR" --iters 100000 --hours "$HOURS" \
  --epochs 2 --minibatch 128 --train gains \
  --eval-every 10 --eval-games 96
