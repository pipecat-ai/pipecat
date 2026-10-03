#!/usr/bin/env bash
# Run the benchmark suite inside a Linux container with explicit CPU/memory
# limits, so results approximate a Kubernetes pod rather than the host laptop.
#
# Why: native macOS uses a different onnxruntime build and a non-glibc allocator,
# so its latency/memory numbers are NOT representative. A Linux container uses the
# prod Linux wheel + glibc and honors MALLOC_ARENA_MAX. Pinning --cpus/--memory to
# the pod's limits reproduces the single biggest production variable (the cgroup
# CPU quota), which sets the throughput plateau and latency-under-load curve.
#
# Usage:
#   ./run_in_docker.sh
#   CPUS=2 MEM=2g ARENA=2 ./run_in_docker.sh          # match your pod's limits
#   MODEL=/path/to/smart-turn-v3.2-cpu.onnx ./run_in_docker.sh
#
# Tunables (env vars, with defaults):
#   CPUS          CPU limit, matches K8s cpu limit           (default 2)
#   MEM           memory limit                                (default 4g)
#                 NOTE: the per-call (status-quo) topology holds N live sessions
#                 (~31 MB each), so the N=64 step needs ~2.5 GB. With a tight
#                 limit (e.g. 2g) that step is OOM-killed - which is itself a
#                 finding: shared sessions reach concurrency that per-call can't.
#                 Set MEM to your real pod limit to see where per-call OOMs.
#   ARENA         MALLOC_ARENA_MAX (prod sets this to 2)      (default 2)
#   ORT_VERSION   onnxruntime version (prod lockfile = 1.24.3)(default 1.24.3)
#   NUMPY_VERSION numpy version                               (default 2.2.6)
#   IMAGE         base image                                  (default python:3.13-slim)
#   MODEL         path to the .onnx model on the host         (auto-detected if unset)
#   CONCURRENCY   latency thread counts                       (default "2 4 8 16 32 64")
#   COUNTS        memory session/in-flight counts             (default "1 2 4 8 16")

set -euo pipefail
cd "$(dirname "$0")"

CPUS="${CPUS:-2}"
MEM="${MEM:-4g}"
ARENA="${ARENA:-2}"
ORT_VERSION="${ORT_VERSION:-1.24.3}"
NUMPY_VERSION="${NUMPY_VERSION:-2.2.6}"
IMAGE="${IMAGE:-python:3.13-slim}"
CONCURRENCY="${CONCURRENCY:-2 4 8 16 32 64}"
COUNTS="${COUNTS:-1 2 4 8 16}"

command -v docker >/dev/null 2>&1 || { echo "docker not found on PATH"; exit 1; }

# --- locate the model on the host ---------------------------------------------
MODEL_PATH="${MODEL:-}"
if [[ -z "$MODEL_PATH" ]]; then
  # 1) a copy sitting next to this script
  if [[ -f "smart-turn-v3.2-cpu.onnx" ]]; then
    MODEL_PATH="$(pwd)/smart-turn-v3.2-cpu.onnx"
  else
    # 2) ask an interpreter that has pipecat installed (try repo venv, then python3)
    for PY in "../../../.venv/bin/python" python3 python; do
      if command -v "$PY" >/dev/null 2>&1 || [[ -x "$PY" ]]; then
        found="$("$PY" - <<'PYEOF' 2>/dev/null || true
from importlib import resources
import os
try:
    p = resources.files("pipecat.audio.turn.smart_turn.data").joinpath("smart-turn-v3.2-cpu.onnx")
    print(p if os.path.exists(str(p)) else "")
except Exception:
    print("")
PYEOF
)"
        if [[ -n "$found" && -f "$found" ]]; then MODEL_PATH="$found"; break; fi
      fi
    done
  fi
fi

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
  echo "Could not locate smart-turn-v3.2-cpu.onnx."
  echo "Set MODEL=/path/to/smart-turn-v3.2-cpu.onnx or copy it next to this script."
  exit 1
fi

echo "Container settings: image=$IMAGE cpus=$CPUS mem=$MEM MALLOC_ARENA_MAX=$ARENA"
echo "onnxruntime==$ORT_VERSION numpy==$NUMPY_VERSION"
echo "Model: $MODEL_PATH"
echo

# --- run ----------------------------------------------------------------------
# Mount the bench scripts read-write (so RESULTS_RAW.txt is written back to the
# host) and the model read-only. cpuset pins to distinct physical CPUs.
# All host values are passed via -e, so the container script below is fully
# single-quoted (no host-shell escaping needed).
docker run --rm \
  --cpus="$CPUS" \
  --memory="$MEM" \
  -e MALLOC_ARENA_MAX="$ARENA" \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -e CONCURRENCY="$CONCURRENCY" \
  -e COUNTS="$COUNTS" \
  -e ORT_VERSION="$ORT_VERSION" \
  -e NUMPY_VERSION="$NUMPY_VERSION" \
  -v "$(pwd)":/bench \
  -v "$MODEL_PATH":/model/smart-turn-v3.2-cpu.onnx:ro \
  -w /bench \
  "$IMAGE" \
  bash -c '
    set -e
    pip install --quiet --no-cache-dir "numpy==${NUMPY_VERSION}" "onnxruntime==${ORT_VERSION}"
    echo "--- container CPU/limit visibility ---"
    python -c "import os; print(\"os.cpu_count():\", os.cpu_count())"
    nproc 2>/dev/null | sed "s/^/nproc: /" || true
    if [ -r /sys/fs/cgroup/cpu.max ]; then
      read -r quota period < /sys/fs/cgroup/cpu.max
      echo "cgroup cpu.max: $quota $period"
      [ "$quota" != "max" ] && awk -v q="$quota" -v p="$period" \
        "BEGIN { printf \"=> effective CPU quota: %.2f cores (os.cpu_count reports host cores, NOT this quota - why intra_op=1 matters)\n\", q/p }"
    fi
    echo
    MODEL=/model/smart-turn-v3.2-cpu.onnx ./run_all.sh python
  '

echo
echo "Done. Raw output (written from the container) is in $(pwd)/RESULTS_RAW.txt"
