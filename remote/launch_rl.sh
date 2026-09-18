#!/bin/bash
# One-shot: point at a running instance, push, start PPO under nohup, start the progress sync.
# Usage: bash remote/launch_rl.sh <instance_id> [hours] [opponent] [envs] [horizon] [tag]
set -e
cd "$(dirname "$0")/.."
ID=$1; HOURS=${2:-2.5}; OPP=${3:-greedy}; ENVS=${4:-96}; HOR=${5:-16}; TAG=${6:-ppo}
.venv/bin/vastai show instance $ID --raw | python3 -c "
import sys,json; d=json.load(sys.stdin)
assert d.get('actual_status') == 'running', d.get('actual_status')
print('machine', d.get('machine_id'), '|', d.get('num_gpus'), d.get('gpu_name'), '| \$%.3f/h' % d['dph_total'], '|', d.get('geolocation'))
open('remote/instance.env','w').write(f\"INSTANCE_ID={d['id']}\nSSH_HOST={d['ssh_host']}\nSSH_PORT={d['ssh_port']}\n\")"
for i in 1 2 3 4 5 6 7 8; do bash remote/ssh.sh "echo ssh ok" 2>/dev/null && break; sleep 10; done
bash remote/push_rl.sh
bash remote/ssh.sh "cd /workspace/fly-chess && (setsid nohup bash remote/rl.sh $HOURS $OPP $ENVS $HOR $TAG > rl.log 2>&1 < /dev/null &) ; sleep 3; pgrep -fl '[r]l.sh' | head -1"
pkill -f "remote/[s]ync" 2>/dev/null || true; (nohup bash remote/sync_rl.sh $TAG > /dev/null 2>&1 &)
echo "launched; sync loop running"
