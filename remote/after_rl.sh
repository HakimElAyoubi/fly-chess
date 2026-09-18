#!/bin/bash
# When PPO finishes, play the same matches with the supervised weights and with the RL weights,
# under identical settings, so the two are comparable. Usage: bash remote/after_rl.sh [tag] [games]
cd "$(dirname "$0")/.."
TAG=${1:-ppo}; GAMES=${2:-400}
while pgrep -f "flybrain[.]rl" > /dev/null; do sleep 60; done
apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y stockfish >/dev/null 2>&1
for OPP in greedy random; do
  python -m flybrain.play --device cuda --opponent $OPP --games $GAMES --concurrency 64 \
    --weights data/train_gpu/model_step11600.pt --tag ${OPP}_baseline
  python -m flybrain.play --device cuda --opponent $OPP --games $GAMES --concurrency 64 \
    --weights data/rl_$TAG/best.pt --tag ${OPP}_rl
done
if command -v stockfish >/dev/null; then
  python -m flybrain.play --device cuda --opponent stockfish --sf-depth 1 --games 200 --concurrency 64 \
    --weights data/train_gpu/model_step11600.pt --tag sfd1_baseline
  python -m flybrain.play --device cuda --opponent stockfish --sf-depth 1 --games 200 --concurrency 64 \
    --weights data/rl_$TAG/best.pt --tag sfd1_rl
fi
echo "=== comparison done"
