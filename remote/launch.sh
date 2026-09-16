#!/bin/bash
# One-shot launch on a running instance: writes remote/instance.env, pushes the code, starts the
# multi-GPU schedule under nohup, and starts the local progress sync. Usage: bash remote/launch.sh <instance_id> <hours_budget>
set -e
cd "$(dirname "$0")/.."
ID=$1; HOURS=${2:-2.8}
.venv/bin/vastai show instance $ID --raw | python3 -c "
import sys,json; d=json.load(sys.stdin)
assert d.get('actual_status') == 'running', d.get('actual_status')
print('machine', d.get('machine_id'), '|', d.get('num_gpus'), d.get('gpu_name'), '| $%.3f/h' % d['dph_total'], '|', d.get('geolocation'))
open('remote/instance.env','w').write(f\"INSTANCE_ID={d['id']}\nSSH_HOST={d['ssh_host']}\nSSH_PORT={d['ssh_port']}\n\")"
for i in 1 2 3 4 5 6 7 8; do bash remote/ssh.sh "echo ssh ok" 2>/dev/null && break; sleep 10; done
bash remote/ssh.sh "nvidia-smi -L | head -8; python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, torch.cuda.is_available())'; mkdir -p /workspace/fly-chess/data" 2>/dev/null
bash remote/push.sh 2>/dev/null && echo "code pushed"
bash remote/ssh.sh "cd /workspace/fly-chess && (setsid nohup bash remote/run_gpu_multi.sh $HOURS > run_gpu.log 2>&1 < /dev/null &) ; sleep 3; pgrep -fl '[r]un_gpu_multi' | head -1" 2>/dev/null
pkill -f "remote/[s]ync.sh" 2>/dev/null; (nohup bash remote/sync.sh > /dev/null 2>&1 &)
echo "launched; sync loop running"
