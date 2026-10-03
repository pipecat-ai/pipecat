"""Latency / throughput: one SHARED session vs N PER-CALL sessions.

For each concurrency level N we spawn N threads that each run inference in a
tight loop, and compare two topologies:

  * shared   : all N threads call Run() on ONE InferenceSession
  * per-call : each thread owns its own InferenceSession (status quo in pipecat,
               where every call constructs its own LocalSmartTurnAnalyzerV3)

If sharing a session introduced contention (arena lock, thread pool), "shared"
would show higher latency / lower throughput than "per-call". We sweep N well
past the core count so any shared-resource contention has room to appear.

The --intra-op sweep separately answers "can we just give the shared session
more threads?": raising intra_op_num_threads helps only at low concurrency and
oversubscribes cores under load.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import common
import numpy as np


def _time_runs(session, x, n: int) -> list[float]:
    out = []
    feed = {common.INPUT_NAME: x}
    for _ in range(n):
        t0 = time.perf_counter()
        session.run(None, feed)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _run_topology(label: str, sessions: list, n_threads: int, reps: int, warmup: int) -> None:
    """sessions: length 1 (shared) or n_threads (per-call)."""
    results: dict[int, list[float]] = {}
    barrier = threading.Barrier(n_threads)

    def worker(tid: int) -> None:
        sess = sessions[0] if len(sessions) == 1 else sessions[tid]
        x = common.make_input(seed=tid)
        _time_runs(sess, x, warmup)  # warm up arena / pages before timing
        barrier.wait()
        results[tid] = _time_runs(sess, x, reps)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    lat = [v for r in results.values() for v in r]
    total = n_threads * reps
    print(
        f"  {label:26s} N={n_threads:3d}  "
        f"median={statistics.median(lat):7.1f}ms  "
        f"p90={np.percentile(lat, 90):7.1f}ms  "
        f"mean={statistics.mean(lat):7.1f}ms  "
        f"throughput={total / wall:6.1f} inf/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--reps", type=int, default=30, help="timed runs per thread")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--intra-op-sweep",
        action="store_true",
        help="also run the intra_op_num_threads comparison",
    )
    args = parser.parse_args()

    model_path = common.resolve_model_path(args.model)
    common.print_environment(model_path)

    print("\nSHARED (1 session, all threads) vs PER-CALL (1 session per thread), intra_op=1")
    print("-" * 72)
    for n in args.concurrency:
        _run_topology(
            "SHARED 1 session",
            [common.make_session(model_path, 1)],
            n,
            args.reps,
            args.warmup,
        )
        _run_topology(
            "PER-CALL N sessions",
            [common.make_session(model_path, 1) for _ in range(n)],
            n,
            args.reps,
            args.warmup,
        )
        print()

    if args.intra_op_sweep:
        print('"More threads on the shared session?" - intra_op_num_threads sweep')
        print("-" * 72)
        low = min(4, (max(args.concurrency)))
        high = max(args.concurrency)
        _run_topology(
            "SHARED intra=4",
            [common.make_session(model_path, 4)],
            low,
            args.reps,
            args.warmup,
        )
        _run_topology(
            "PER-CALL intra=4",
            [common.make_session(model_path, 4) for _ in range(low)],
            low,
            args.reps,
            args.warmup,
        )
        _run_topology(
            "SHARED intra=4 (overload)",
            [common.make_session(model_path, 4)],
            high,
            args.reps,
            args.warmup,
        )
        _run_topology(
            "SHARED intra=1 (overload)",
            [common.make_session(model_path, 1)],
            high,
            args.reps,
            args.warmup,
        )


if __name__ == "__main__":
    main()
