#!/bin/bash
# Self-refreshing progress bar in the terminal (Ctrl-C to stop). The live page is http://127.0.0.1:8931/progress.html
cd "$(dirname "$0")"
while :; do clear; echo "Fly Chess · progress · $(date +%H:%M:%S)"; echo; cat data/progress.txt 2>/dev/null || echo "no job running"; sleep 2; done
