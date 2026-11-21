# Results: sharing one ONNX `InferenceSession` across smart-turn calls

**TL;DR — sharing a single `InferenceSession` across all calls is correct, has no
measurable latency/throughput penalty (even at 5× the core count), and turns a
per-call memory cost of ~31 MB into a one-time fixed cost. It is a safe,
high-value optimization, and is what pipecat now does by default.**

The raw, unedited console capture is in [RESULTS_RAW.txt](RESULTS_RAW.txt). The
tables below are transcribed from a representative run. These numbers exercise the
**smart-turn** session; `SileroVADAnalyzer` shares its session through the same
mechanism and is covered by the unit tests rather than this benchmark (see the
README "Scope" section).

## Test machine

| | |
|---|---|
| platform | macOS 15.7.7, x86_64 |
| logical CPUs | 12 |
| Python | 3.13.9 |
| onnxruntime | 1.24.4 (CPUExecutionProvider) |
| model | `smart-turn-v3.2-cpu.onnx`, 8.28 MB on disk |
| session options | `ORT_SEQUENTIAL`, `inter_op=1`, `intra_op=1`, `ORT_ENABLE_ALL` (pipecat defaults) |

> These numbers are from native macOS and are used only to establish the
> *relative* conclusions, which are platform independent. Absolute MB/ms are not
> production-representative (macOS uses a different onnxruntime build and a
> non-glibc allocator, and ignores `MALLOC_ARENA_MAX`). For prod-like numbers run
> `./run_in_docker.sh` with your pod's `CPUS`/`MEM` limits, or run in an actual
> pod — see the README.

## 1. Correctness — concurrent `Run()` on a shared session is thread-safe

```
Concurrent runs : 3200 (16 threads x 200 reps over 4 inputs)
RESULT: PASS - every concurrent output bit-for-bit matched its single-threaded reference.
```

16 threads hammered a single shared session with 3,200 interleaved inferences;
every output was bit-for-bit identical to the single-threaded reference for the
same input. This matches ONNX Runtime's documented contract: `Run()` is
thread-safe for concurrent calls on one session (CPU EP), and `Run()` releases
the GIL so the Python threads execute in parallel.

## 2. Latency / throughput — shared vs per-call

N concurrent threads, each looping inference; `SHARED` = one session for all,
`PER-CALL` = one session per thread (the previous behavior). `intra_op=1`.

| N | SHARED median / p90 / thru | PER-CALL median / p90 / thru |
|---:|---|---|
| 2  | 92.9 / 95.0 ms / 18.2 | 94.6 / 98.3 ms / 17.9 |
| 4  | 94.8 / 102.4 ms / 34.9 | 98.8 / 115.3 ms / 33.3 |
| 8  | 160.2 / 173.8 ms / 40.9 | 162.1 / 172.6 ms / 41.8 |
| 16 | 334.9 / 385.6 ms / 39.4 | 349.1 / 373.1 ms / 38.9 |
| 32 | 695.5 / 762.4 ms / 38.7 | 713.4 / 776.1 ms / 37.8 |
| 64 | 1461.6 / 1761.6 ms / 36.5 | 1614.5 / 1911.2 ms / 33.4 |

**Shared tracks per-call within run-to-run noise across the entire sweep — up to
64 concurrent inferences, 5× the 12 cores.** There is no contention penalty from
the shared session's memory arena or internals. Both topologies are CPU-bound:
throughput plateaus near the core count (~40 inf/s) and latency past N≈12 grows
purely from queuing for cores — identically for both. At N=64 the shared session
is consistently *faster* than per-call, because 64 independent sessions add
allocator/working-set pressure that one shared session avoids.

## 3. "Can we just give the shared session more threads?" — `intra_op` sweep

| config | N | median | thru |
|---|---:|---:|---:|
| SHARED `intra=4` | 4 | 107.9 ms | 30.7 |
| PER-CALL `intra=4` | 4 | 174.0 ms | 19.3 |
| SHARED `intra=4` (overload) | 64 | 1537 ms | 33.1 |
| SHARED `intra=1` (overload) | 64 | 1684 ms | 30.1 |

- **Raising `intra_op` does not buy throughput here.** Compared with the
  `intra=1` baseline (N=4 SHARED ≈ 95 ms above), `intra=4` is no better — and on
  this run slightly worse — because the work is already parallel across calls.
- **Per-call sessions at `intra=4` are clearly worse** (174 ms vs ~95 ms):
  4 sessions × 4 threads = 16 threads oversubscribe 12 cores. A single shared
  session bounds this to one thread pool, so it degrades far less.
- **At high concurrency** thread count is in the noise (1537 vs 1684 ms) — the
  parallelism comes from many calls in flight, not from intra-op threads.

**Conclusion:** the existing design — `intra_op=1` plus one executor thread per
call against a shared session — is already the right configuration for a
many-calls server. If a hard ceiling on total ORT threads is ever desired, use a
global thread pool (`SessionOptions.use_per_session_threads=False` +
`onnxruntime.set_global_thread_pool_sizes(...)`), not per-session thread counts.

## 4. Memory — the win (measured in isolated processes)

### A. Per-call: N independent sessions (previous behavior)

| sessions | RSS | over baseline | per session |
|---:|---:|---:|---:|
| baseline | 42.9 MB | — | — |
| 1 | 96.9 MB | +54.0 | 54.0 (incl. one-time ORT init) |
| 2 | 136.8 MB | +93.9 | 47.0 |
| 4 | 197.6 MB | +154.7 | 38.7 |
| 8 | 309.1 MB | +266.3 | 33.3 |
| 16 | 532.5 MB | +489.6 | **30.6** |

Each additional call's session costs ~31 MB steady-state (the first is higher
because it includes ORT's one-time runtime init). For an 8.3 MB model this
overhead is pre-packed weights + graph + per-session memory arena.

### B. Shared: one session under N *simultaneous* inferences

| state | RSS | over baseline/idle |
|---|---:|---:|
| baseline | 43.0 MB | — |
| 1 shared session, idle | 86.8 MB | **+43.8 (fixed)** |
| 1 in-flight | 96.5 MB | +9.8 |
| 2 in-flight | 111.0 MB | +24.2 |
| 4 in-flight | 143.9 MB | +57.2 |
| 8 in-flight | 217.4 MB | +130.6 |
| 16 in-flight | 380.0 MB | +293.3 (~18.3 / in-flight) |

The shared session has a **fixed ~44 MB cost** (weights + pre-packed weights +
graph + ORT init, loaded once) plus arena activations of **~15–18 MB per
inference that is in flight at the same instant**. Crucially, that arena scales
with *simultaneous* inferences, not with the number of calls.

> The shared BFCArena grows to its concurrency high-water mark and does not
> shrink by default. If that high-water RSS matters, set
> `enable_cpu_mem_arena=False` (slightly slower, returns memory per run) or
> trigger per-run arena shrinkage. Likely unnecessary.

### What this means at scale

```
per-call total   ≈ (concurrent CALLS)               × ~31 MB
shared total     ≈ ~44 MB  +  (SIMULTANEOUS inferences) × ~18 MB
```

Smart-turn runs only at end-of-turn (a ~90 ms burst per utterance), so the number
of inferences in flight at any instant is far smaller than the number of active
calls. Illustratively, for **100 concurrent calls** with, say, ~8 inferences ever
overlapping:

- per-call: ~3.1 GB resident, all the time.
- shared: ~44 MB + 8 × 18 MB ≈ **~190 MB**.

A **>10×** reduction, with no performance cost.

## 5. Container / pod observations (`run_in_docker.sh`)

Running the suite in `python:3.13-slim` on a Linux VM, limited to `--cpus=2`,
surfaced two production-relevant points (raw: a Linux run shows
`platform: Linux-...-with-glibc2.41`, `onnxruntime 1.24.3`):

- **`os.cpu_count()` reported 12 while the cgroup `cpu.max` was `200000 100000`
  (a 2-core quota).** ONNX Runtime's default thread-pool sizing reads the host
  core count, not the cgroup quota, so an unpinned session in a CPU-limited pod
  would spawn a 12-wide pool and then be CFS-throttled to 2 cores. The smart-turn
  path pins `intra_op=1`, which sidesteps this — a concrete reason to keep
  `intra_op` explicit and not let it default.
- **The per-call topology was OOM-killed at `--memory=2g`** at N=64 (64 live
  sessions ≈ 2.5 GB), while the shared session needs only ~0.4 GB at the same
  concurrency. So in a memory-constrained pod the duplication is not merely
  wasteful — it caps the concurrency the previous behavior could reach at all.
  Sharing removes that ceiling. (The suite isolates steps so an OOM in one does
  not abort the others.)

Absolute latency/throughput numbers from a 2-core container will be much lower
than the 12-core figures in §2–3; that is expected and is why capacity numbers
must come from a correctly-limited environment, ideally a real pod.

## What pipecat implements

`SileroVADAnalyzer` and `LocalSmartTurnAnalyzerV3` obtain their session from a
process-wide cache keyed by model path + options (`cpu_count` for smart-turn,
`force_onnx_cpu` for Silero). **Everything else stays per-instance:**

- All per-turn state (`_audio_buffer`, `_speech_triggered`, `_silence_ms`,
  `_speech_start_time`, `_sample_rate`, `_vad_start_secs`) stays on the analyzer
  instance — it never touched the session, so sharing changes nothing there.
- The Silero recurrent state (`_state`/`_context`) stays on each `SileroOnnxModel`
  instance and is reset per instance.
- The per-instance `ThreadPoolExecutor(max_workers=1)` stays per-instance, so
  each call's inference still runs on its own thread against the shared session —
  which is exactly what makes `intra_op=1` give N-way parallelism.
- The analyzer object is **not** shared; `intra_op` is **not** raised globally.

The change was validated downstream in production before being upstreamed.

## Reproduce

```bash
./run_all.sh /path/to/python      # or: MODEL=/path/to/model.onnx ./run_all.sh
```

See [README.md](README.md) for per-script usage and flags.

## Sources

- ONNX Runtime — Thread management / threading docs:
  <https://onnxruntime.ai/docs/performance/tune-performance/threading.html>
- "Is `InferenceSession.Run` thread safe?" — microsoft/onnxruntime#114 and
  Discussion #10107 (concurrent `Run()` on one session is thread-safe for the
  CPU EP; session construction is serialized and holds the GIL).
