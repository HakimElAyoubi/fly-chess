#!/bin/bash
# Code plus the three data files the RL job needs (graph, config, supervised weights).
source "$(dirname "$0")/instance.env"
cd "$(dirname "$0")/.."
SCP="scp -q -i $HOME/.ssh/vast_softgroup -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $SSH_PORT"
rsync -az -e "ssh -i $HOME/.ssh/vast_softgroup -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT" \
  --exclude '.venv' --exclude 'data' --exclude '__pycache__' --exclude '*.html' --exclude '.git' --exclude 'remote/instance.env' \
  ./ "root@$SSH_HOST:/workspace/fly-chess/"
bash remote/ssh.sh "mkdir -p /workspace/fly-chess/data/train_gpu"
$SCP data/graph_brain.npz data/graph_brain_meta.feather data/graph_brain_summary.json data/checks.json "root@$SSH_HOST:/workspace/fly-chess/data/"
$SCP data/train_gpu/model_step11600.pt "root@$SSH_HOST:/workspace/fly-chess/data/train_gpu/"
echo "code and data pushed"
