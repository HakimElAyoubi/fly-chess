#!/bin/bash
# Runs on the laptop: every 5 minutes check the Vast credit; below the floor, collect results and
# end the training so the watchdog stops the box with a safe margin. Usage: bash remote/credit_guard.sh <floor_dollars>
FLOOR=${1:-1.6}
cd "$(dirname "$0")/.."
source remote/instance.env
while :; do
  C=$(.venv/bin/vastai show user --raw 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['credit'])" 2>/dev/null)
  RUNNING=$(.venv/bin/vastai show instance $INSTANCE_ID --raw 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status'))" 2>/dev/null)
  echo "$(date -u +%H:%M) credit \$$C instance $RUNNING" >> data/credit_guard.log
  [ "$RUNNING" != "running" ] && exit 0
  if [ -n "$C" ] && python3 -c "import sys; sys.exit(0 if float('$C') < $FLOOR else 1)"; then
    echo "credit below floor: collecting and ending the run" >> data/credit_guard.log
    bash remote/collect.sh >> data/credit_guard.log 2>&1
    bash remote/ssh.sh "pkill -f torchrun; pkill -f 'flybrain.train'; sleep 5; pkill -f run_gpu_multi" >> data/credit_guard.log 2>&1
    exit 0
  fi
  sleep 300
done
