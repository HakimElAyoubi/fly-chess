#!/bin/bash
# Stop this instance when the job is genuinely over, so GPU billing ends without needing the
# laptop. Deliberately conservative: it requires BOTH that no training process is alive AND that
# the log has stopped changing for a while, because a watchdog that fires early costs a whole run.
# Usage: bash remote/selfstop_watch.sh [log] [idle_minutes]
cd /workspace/fly-chess
LOG=${1:-run_gpu.log}; IDLE=${2:-20}
log() { echo "$(date -u +%H:%M:%S) $*" >> selfstop.log; }
log "watchdog armed on $LOG, idle threshold ${IDLE}m"
sleep 300
while :; do
  BUSY=$(pgrep -f "torchrun|flybrain\.(train|label|data|probe)" | wc -l)
  AGE=$(( ($(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || date +%s)) / 60 ))
  if [ "$BUSY" -eq 0 ] && [ "$AGE" -ge "$IDLE" ]; then break; fi
  sleep 60
done
log "no training process and log idle ${AGE}m; waiting up to 20m for the results to be copied"
for i in $(seq 1 40); do [ -f COPIED ] && break; sleep 30; done
log "stopping instance (COPIED: $([ -f COPIED ] && echo yes || echo no))"
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(ID|API_KEY)=' | sed 's/^/export /')"
if [ -n "$CONTAINER_ID" ] && [ -n "$CONTAINER_API_KEY" ]; then
  curl -s -X PUT "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" \
    -H "Authorization: Bearer $CONTAINER_API_KEY" -H "Content-Type: application/json" \
    -d '{"state":"stopped"}' >> selfstop.log 2>&1
  log "self-stop requested"
else
  log "no container credentials; cannot self-stop"
fi
