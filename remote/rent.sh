#!/bin/bash
# Rent the first offer that actually reaches "running". Failed contracts are destroyed.
# Usage: bash remote/rent.sh <image> <offer_id> [offer_id ...]
cd "$(dirname "$0")/.."
IMAGE=$1; shift
for OFFER in "$@"; do
  echo "trying offer $OFFER"
  R=$(.venv/bin/vastai create instance "$OFFER" --image "$IMAGE" --disk 100 --ssh --direct --label fly-chess --raw 2>&1)
  ID=$(echo "$R" | python3 -c "import sys,json; print(json.load(sys.stdin).get('new_contract',''))" 2>/dev/null)
  OK=$(echo "$R" | grep -c '"success": true')
  if [ "$OK" != "1" ] || [ -z "$ID" ]; then
    echo "  create refused"; [ -n "$ID" ] && (yes | .venv/bin/vastai destroy instance "$ID" >/dev/null 2>&1); continue
  fi
  for i in $(seq 1 60); do
    J=$(.venv/bin/vastai show instance "$ID" --raw 2>/dev/null)
    ST=$(echo "$J" | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status'))" 2>/dev/null)
    MSG=$(echo "$J" | python3 -c "import sys,json; print((json.load(sys.stdin).get('status_msg') or '')[:70].replace(chr(10),' '))" 2>/dev/null)
    [ "$ST" = "running" ] && break
    echo "$MSG" | grep -qi "error\|failed\|unavailable" && { ST=broken; break; }
    sleep 15
  done
  if [ "$ST" = "running" ]; then
    echo "$J" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print('RUNNING', d['id'], d['num_gpus'], d['gpu_name'], '\$%.3f/h' % d['dph_total'], d.get('geolocation'))
open('remote/instance.env','w').write(f\"INSTANCE_ID={d['id']}\nSSH_HOST={d['ssh_host']}\nSSH_PORT={d['ssh_port']}\n\")"
    exit 0
  fi
  echo "  offer $OFFER did not come up ($ST); destroying"
  yes | .venv/bin/vastai destroy instance "$ID" >/dev/null 2>&1
done
echo "no offer came up"; exit 1
