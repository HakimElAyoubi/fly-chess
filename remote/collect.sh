#!/bin/bash
# Copy the run's log and latest checkpoint back from the box, then leave the COPIED marker so
# the watchdog may stop the instance. Usage: bash remote/collect.sh
source "$(dirname "$0")/instance.env"
cd "$(dirname "$0")/.."
SSH="ssh -i $HOME/.ssh/vast_softgroup -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT"
mkdir -p data/train_gpu
scp -q -i "$HOME/.ssh/vast_softgroup" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$SSH_PORT" "root@$SSH_HOST:/workspace/fly-chess/run_gpu.log" data/run_gpu.log
scp -q -i "$HOME/.ssh/vast_softgroup" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$SSH_PORT" "root@$SSH_HOST:/workspace/fly-chess/data/train_gpu/log.jsonl" data/train_gpu/log.jsonl
$SSH "root@$SSH_HOST" "cp /workspace/fly-chess/data/train_gpu/checkpoint.pt /workspace/fly-chess/data/train_gpu/checkpoint_copy.pt"
scp -q -i "$HOME/.ssh/vast_softgroup" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$SSH_PORT" "root@$SSH_HOST:/workspace/fly-chess/data/train_gpu/checkpoint_copy.pt" data/train_gpu/checkpoint.pt
.venv/bin/python -c "import torch; ck=torch.load('data/train_gpu/checkpoint.pt', map_location='cpu', weights_only=False); print('checkpoint ok, step', ck['step'])" && $SSH "root@$SSH_HOST" "touch /workspace/fly-chess/COPIED" && echo "COPIED marker set"
