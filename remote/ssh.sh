#!/bin/bash
# ssh to the current Vast.ai instance. Endpoint in remote/instance.env (INSTANCE_ID, SSH_HOST, SSH_PORT).
source "$(dirname "$0")/instance.env"
exec ssh -i "$HOME/.ssh/vast_softgroup" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -p "$SSH_PORT" "root@$SSH_HOST" "$@"
