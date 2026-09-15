#!/bin/bash
# copy the code (not the data) to the instance
source "$(dirname "$0")/instance.env"
cd "$(dirname "$0")/.."
rsync -az -e "ssh -i $HOME/.ssh/vast_softgroup -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT" \
  --exclude '.venv' --exclude 'data' --exclude '__pycache__' --exclude '*.html' --exclude '.git' --exclude 'remote/instance.env' \
  ./ "root@$SSH_HOST:/workspace/fly-chess/"
