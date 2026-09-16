#!/bin/bash
# pull the GPU job's progress bar and log every 15 s into data/ (feeds progress.html)
source "$(dirname "$0")/instance.env"
cd "$(dirname "$0")/.."
while :; do
  scp -q -i "$HOME/.ssh/vast_softgroup" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -P "$SSH_PORT" \
    "root@$SSH_HOST:/workspace/fly-chess/data/progress.txt" data/progress_gpu.txt 2>/dev/null
  scp -q -i "$HOME/.ssh/vast_softgroup" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -P "$SSH_PORT" \
    "root@$SSH_HOST:/workspace/fly-chess/run_gpu.log" data/run_gpu.log 2>/dev/null
  scp -q -i "$HOME/.ssh/vast_softgroup" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -P "$SSH_PORT" \
    "root@$SSH_HOST:/workspace/fly-chess/data/train_gpu/log.jsonl" data/train_gpu/log_remote.jsonl 2>/dev/null
  sleep 15
done
