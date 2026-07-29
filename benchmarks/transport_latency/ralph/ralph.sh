#!/usr/bin/env bash
# Ralph loop for Phase B of the transport latency benchmark.
#
# Runs one `claude -p` iteration per unchecked task in checklist.md until the
# list is done or an iteration files a human gate in HUMAN_TODO.md. Iterations
# never commit — review and commit between runs if you want checkpoints.
#
# Usage (from anywhere):  benchmarks/transport_latency/ralph/ralph.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"

while true; do
    if grep -q '^## PENDING' "$HERE/HUMAN_TODO.md" 2>/dev/null; then
        echo
        echo "ralph: paused — human action needed. See:"
        echo "  $HERE/HUMAN_TODO.md"
        echo "Mark the section '## DONE:' (or delete it) and rerun ralph.sh."
        exit 1
    fi
    if ! grep -q '^- \[ \]' "$HERE/checklist.md"; then
        echo "ralph: checklist complete."
        exit 0
    fi
    next_task="$(grep -m1 '^- \[ \]' "$HERE/checklist.md" | cut -c7- | cut -d. -f1)"
    echo "ralph: starting iteration — $next_task"
    claude -p "$(cat "$HERE/PROMPT.md")" --permission-mode acceptEdits
done
