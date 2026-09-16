#!/bin/bash
# Runs on the box: when the schedule process is gone (finished or crashed), wait up to 20 minutes
# for a COPIED marker (the laptop copying results back), then stop this instance so GPU billing
# ends without needing the laptop. Credentials come from the container's init env.
cd /workspace/fly-chess
sleep 120
while pgrep -f "[r]un_gpu_multi.sh" > /dev/null; do sleep 30; done
echo "schedule process gone at $(date -u)" >> selfstop.log
for i in $(seq 1 40); do [ -f COPIED ] && break; sleep 30; done
echo "stopping instance at $(date -u) (COPIED marker: $([ -f COPIED ] && echo yes || echo no))" >> selfstop.log
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(ID|API_KEY)=' | sed 's/^/export /')"
if [ -n "$CONTAINER_ID" ] && [ -n "$CONTAINER_API_KEY" ]; then
  curl -s -X PUT "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" -H "Authorization: Bearer $CONTAINER_API_KEY" -H "Content-Type: application/json" -d '{"state":"stopped"}' >> selfstop.log 2>&1
  echo " self-stop requested" >> selfstop.log
else
  echo "no container credentials found" >> selfstop.log
fi
