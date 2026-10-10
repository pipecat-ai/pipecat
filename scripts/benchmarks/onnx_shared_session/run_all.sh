#!/usr/bin/env bash
# Run the full smart-turn shared-session benchmark suite and tee to RESULTS_RAW.txt.
#
# Usage:
#   ./run_all.sh [PYTHON]
# where PYTHON defaults to `python`. Example with a project venv:
#   ./run_all.sh /path/to/.venv/bin/python
#
# Env overrides (passed through to the scripts):
#   CONCURRENCY   latency/throughput thread counts   (default "2 4 8 16 32 64")
#   COUNTS        memory session/in-flight counts     (default "1 2 4 8 16")
#
# Requires: numpy, onnxruntime, and either pipecat-ai installed (to locate the
# bundled model) or set MODEL=/path/to/smart-turn-v3.2-cpu.onnx.
#
# Steps are run independently: if one is killed (e.g. the OOM killer terminates
# the per-call high-concurrency step under a tight memory limit) the remaining
# steps still run, and the failure is reported.

cd "$(dirname "$0")"

PY="${1:-python}"
MODEL_ARG=()
if [[ -n "${MODEL:-}" ]]; then
  MODEL_ARG=(--model "$MODEL")
fi
CONCURRENCY="${CONCURRENCY:-2 4 8 16 32 64}"
COUNTS="${COUNTS:-1 2 4 8 16}"

run_step() {
  local title="$1"; shift
  echo "########################################################################"
  echo "# $title"
  echo "########################################################################"
  "$@"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo ">>> STEP FAILED (exit $rc). If exit is 137/Killed this is almost"
    echo ">>> certainly the OOM killer: the per-call topology needs more memory"
    echo ">>> than the limit. Raise --memory, lower CONCURRENCY, or note that the"
    echo ">>> status-quo (per-call) approach cannot reach this concurrency here."
  fi
  echo
  return 0
}

OUT=RESULTS_RAW.txt
{
  run_step "1/3 CORRECTNESS - concurrent Run() on a shared session is thread-safe" \
    "$PY" correctness.py ${MODEL_ARG[@]+"${MODEL_ARG[@]}"}

  run_step "2/3 LATENCY / THROUGHPUT - shared vs per-call, concurrency sweep" \
    "$PY" latency_throughput.py ${MODEL_ARG[@]+"${MODEL_ARG[@]}"} \
      --concurrency $CONCURRENCY --intra-op-sweep

  run_step "3/3 MEMORY - per-call cost vs shared fixed+arena cost" \
    "$PY" memory.py ${MODEL_ARG[@]+"${MODEL_ARG[@]}"} --counts $COUNTS
} 2>&1 | tee "$OUT"

echo
echo "Wrote raw output to $(pwd)/$OUT"
