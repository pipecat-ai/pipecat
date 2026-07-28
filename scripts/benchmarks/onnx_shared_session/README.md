# Smart-turn ONNX: shared `InferenceSession` benchmark

Before this change, `LocalSmartTurnAnalyzerV3` and `SileroVADAnalyzer` each built a
**new ONNX `InferenceSession` per instance**. In a server that handles many
simultaneous calls, every call builds its own analyzer and therefore loads its
own copy of the model into memory — even though the model weights are identical
and the session is used in a purely stateless way.

Pipecat now caches the session process-wide (keyed by model path + options), so
all analyzer instances of the same configuration share one session. This suite is
the evidence behind that change. It answers three questions:

1. **Correctness** — is concurrent `Run()` on one shared session thread-safe?
2. **Performance** — does sharing add latency or reduce throughput vs per-call
   sessions, including past the core count?
3. **Memory** — how much does per-call session duplication actually cost, and
   what does sharing save?

It also answers the natural follow-up: *"can we just give the shared session
more intra-op threads instead?"*

## Scope: what the benchmark exercises

These scripts drive the **smart-turn** session directly (raw ONNX Runtime, with
the exact `SessionOptions` pipecat uses). `SileroVADAnalyzer` shares its session
through the same mechanism and rests on the same argument — the session is
immutable and stateless, all recurrent VAD state lives on the Python wrapper — but
that case is validated by the unit tests (`tests/test_onnx_session_sharing.py`),
not by this benchmark.

## What is and isn't shared

Only the read-only `InferenceSession` (graph + weights) is the duplicated cost.
All per-turn mutable state in pipecat's `BaseSmartTurn` — `_audio_buffer`,
`_speech_triggered`, `_silence_ms`, `_speech_start_time`, `_sample_rate`,
`_vad_start_secs`, and the per-instance `ThreadPoolExecutor(max_workers=1)` —
lives on the analyzer instance, not on the session. The implemented change keeps
one analyzer instance per call and shares **only** `self._session`, so lifecycle
hooks (`set_sample_rate`, `setup`/`start`/`stop`) keep mutating per-call state.

## Requirements

- Python 3.10+
- `numpy`, `onnxruntime` (CPU)
- The model file `smart-turn-v3.2-cpu.onnx`. The scripts locate it automatically
  if `pipecat-ai` is installed; otherwise pass `--model /path/to/the.onnx` (or set
  `MODEL=...` for `run_all.sh`).

The scripts have **no dependency on pipecat application code** — they only need
the model file — so they are self-contained.

## Run everything

```bash
./run_all.sh                      # uses `python` on PATH
./run_all.sh /path/to/venv/bin/python
MODEL=/path/to/smart-turn-v3.2-cpu.onnx ./run_all.sh
```

Raw output is teed to `RESULTS_RAW.txt`. A captured run with analysis is in
[RESULTS.md](RESULTS.md).

## Running under realistic CPU/memory limits (Linux container)

Native macOS numbers are **not representative**: macOS uses a different
onnxruntime build and a non-glibc allocator, and ignores `MALLOC_ARENA_MAX`
(which production typically sets to `2`). Run inside a Linux container instead,
pinned to the same CPU/memory limits as the target Kubernetes pod — the cgroup
CPU quota is the single biggest factor in the latency/throughput curve.

```bash
# Defaults: python:3.13-slim, --cpus=2 --memory=4g, MALLOC_ARENA_MAX=2,
# onnxruntime==1.24.3, numpy==2.2.6.
./run_in_docker.sh

# Match your actual pod limits:
CPUS=4 MEM=3g ARENA=2 ./run_in_docker.sh

# Point at a specific model / onnxruntime version:
MODEL=/path/to/smart-turn-v3.2-cpu.onnx ORT_VERSION=1.24.3 ./run_in_docker.sh
```

The wrapper mounts this folder into the container (so `RESULTS_RAW.txt` is
written back to the host), mounts the model read-only, prints what the container
sees for `os.cpu_count()` and the cgroup `cpu.max`, then runs `run_all.sh`. It
auto-detects the model from an installed pipecat (or set `MODEL`).

**What transfers from a correctly-limited container**: correctness,
shared ≈ per-call (no regression), and the memory decomposition.
**What still needs a real pod**: absolute p90 latency, the throughput plateau for
capacity planning, and behavior under memory-cgroup pressure / noisy neighbors.
On Apple Silicon hosts, `linux/amd64` images run under emulation (slow,
unrepresentative); use an `arm64` runner or a real pod for timing there.

## Run individually

```bash
# 1. Thread-safety proof: concurrent outputs must match single-threaded refs.
python correctness.py --threads 16 --reps 200 --inputs 4

# 2. Shared vs per-call latency/throughput across a concurrency sweep,
#    plus the intra_op_num_threads comparison.
python latency_throughput.py --concurrency 2 4 8 16 32 64 --intra-op-sweep

# 3. Per-call session RSS vs shared fixed + per-in-flight arena RSS.
python memory.py --counts 1 2 4 8 16
```

All scripts accept `--model`. See `--help` on each for the full flag list.

## How to read the results

- **correctness** prints `PASS`/`FAIL`. PASS means every output produced under
  concurrent load was bit-for-bit identical to the single-threaded reference for
  the same input — i.e. no data race inside the shared session.
- **latency_throughput** prints, per concurrency level `N`, the `SHARED`
  (one session) and `PER-CALL` (N sessions) rows. If `SHARED` matches `PER-CALL`,
  sharing has no contention cost. Throughput plateaus at the core count because
  the workload is CPU-bound; latency grows with `N` past that purely from
  queuing for cores — identically for both topologies.
- **memory** shows per-call RSS scaling with the *number of sessions*, versus the
  shared session's fixed cost plus arena that scales only with the *number of
  inferences in flight at once*. Subtract the printed baseline to isolate the
  ONNX cost.

## Files

| file | purpose |
|------|---------|
| `common.py` | model location, session factory (matches pipecat's options), RSS helper |
| `correctness.py` | concurrent-vs-reference output equality (thread-safety proof) |
| `latency_throughput.py` | shared vs per-call sweep + intra-op sweep |
| `memory.py` | per-call vs shared memory decomposition |
| `run_all.sh` | runs all three, tees to `RESULTS_RAW.txt` |
| `run_in_docker.sh` | runs the suite in a Linux container with pod-like CPU/mem limits |
| `RESULTS.md` | a captured run with analysis and conclusion |
| `RESULTS_RAW.txt` | unedited console capture from a representative run |
