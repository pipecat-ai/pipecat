"""Memory: where the per-call cost goes, and what sharing actually saves.

Two measurements:

  A. PER-CALL cost (--mode per-call) — create N independent sessions (each
     warmed with one inference) and report resident set size after each step.
     This is the status quo: every concurrent call holds its own session for
     the whole call.

  B. SHARED cost (--mode shared) — build ONE session, then drive N *simultaneous*
     inferences through it. This shows the honest decomposition:
        fixed   = weights + pre-packed weights + graph (shared once)
        scaling = per-inference arena activations, which scale with the number
                  of inferences IN FLIGHT AT ONCE, not with the call count.

The two modes MUST run in separate processes: ONNX Runtime / the C allocator do
not return freed session memory to the OS, so measuring "shared" after building
and dropping 16 per-call sessions would report a polluted baseline. With no
--mode (the default) this script re-executes itself once per mode in a fresh
subprocess so each starts from a clean baseline.

RSS is process resident memory (MiB) via `ps`. Numbers include the Python +
onnxruntime + numpy baseline, which we print first so you can subtract it.
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import threading

import common


def measure_per_call(model_path: str, counts: list[int]) -> None:
    print("\nA. PER-CALL: N independent sessions (status quo)")
    print("-" * 72)
    base = common.current_rss_mb()
    print(f"  baseline (no session)         RSS={base:7.1f} MB")
    sessions = []
    x = common.make_input(seed=0)
    made = 0
    for n in sorted(counts):
        while made < n:
            s = common.make_session(model_path, intra_op=1)
            s.run(None, {common.INPUT_NAME: x})  # warm: allocate arena
            sessions.append(s)
            made += 1
        gc.collect()
        rss = common.current_rss_mb()
        print(
            f"  {n:3d} sessions (warmed)         RSS={rss:7.1f} MB  "
            f"(+{rss - base:7.1f} over base, ~{(rss - base) / n:5.1f} MB/session)"
        )


def measure_shared(model_path: str, counts: list[int]) -> None:
    print("\nB. SHARED: 1 session under N simultaneous inferences")
    print("-" * 72)
    base = common.current_rss_mb()
    print(f"  baseline (no session)         RSS={base:7.1f} MB")
    session = common.make_session(model_path, intra_op=1)
    x = common.make_input(seed=0)
    session.run(None, {common.INPUT_NAME: x})
    gc.collect()
    idle = common.current_rss_mb()
    print(
        f"  1 shared session, idle arena  RSS={idle:7.1f} MB  "
        f"(+{idle - base:7.1f} over base = fixed cost)"
    )

    feed = {common.INPUT_NAME: x}

    def hammer(reps: int) -> None:
        for _ in range(reps):
            session.run(None, feed)

    for n in sorted(counts):
        threads = [threading.Thread(target=hammer, args=(40,)) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        gc.collect()
        rss = common.current_rss_mb()
        print(
            f"  {n:3d} simultaneous inferences   RSS={rss:7.1f} MB  "
            f"(+{rss - idle:7.1f} over idle, ~{(rss - idle) / n:5.1f} MB/in-flight)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--mode",
        choices=["per-call", "shared"],
        default=None,
        help="run a single measurement (used internally for "
        "process isolation); default runs both in fresh "
        "subprocesses",
    )
    args = parser.parse_args()

    model_path = common.resolve_model_path(args.model)

    if args.mode == "per-call":
        measure_per_call(model_path, args.counts)
        return
    if args.mode == "shared":
        measure_shared(model_path, args.counts)
        return

    # Default: print env once, then run each mode in its own process so the
    # second mode is not biased by retained allocations from the first.
    common.print_environment(model_path)
    sys.stdout.flush()  # flush before subprocesses write to the shared stdout fd
    counts = [str(c) for c in args.counts]
    base = [sys.executable, __file__, "--counts", *counts]
    if args.model:
        base += ["--model", args.model]
    subprocess.run(base + ["--mode", "per-call"], check=True)
    subprocess.run(base + ["--mode", "shared"], check=True)
    print("\nInterpretation:")
    print("  per-call total memory  ~= (concurrent CALLS)        x MB/session")
    print("  shared total memory    ~= fixed + (SIMULTANEOUS inferences) x MB/in-flight")
    print("  Smart-turn fires only at end-of-turn (~tens of ms), so simultaneous")
    print("  inferences are far fewer than concurrent calls -> large saving.")


if __name__ == "__main__":
    main()
