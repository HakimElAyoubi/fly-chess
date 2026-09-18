#!/bin/bash
# pull the RL job's progress bar and learning curve every 15 s (feeds progress.html)
source "$(dirname "$0")/instance.env"
cd "$(dirname "$0")/.."
TAG=${1:-ppo}
mkdir -p "data/rl_$TAG"
S="-q -i $HOME/.ssh/vast_softgroup -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -P $SSH_PORT"
while :; do
  scp $S "root@$SSH_HOST:/workspace/fly-chess/data/progress.txt" data/progress_gpu.txt 2>/dev/null
  scp $S "root@$SSH_HOST:/workspace/fly-chess/rl.log" data/rl.log 2>/dev/null
  scp $S "root@$SSH_HOST:/workspace/fly-chess/data/rl_$TAG/log.jsonl" "data/rl_$TAG/log_remote.jsonl" 2>/dev/null
  sleep 15
done
