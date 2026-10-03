"""Correctness: concurrent Run() on ONE shared session is thread-safe.

This is the load-bearing claim behind sharing a single InferenceSession across
calls. ONNX Runtime documents Run() as thread-safe for the CPU EP
(https://onnxruntime.ai/docs/performance/tune-performance/threading.html), and
this test demonstrates it empirically:

  1. Compute a reference output for each of K fixed inputs, single-threaded.
  2. Hammer the SAME session from N threads, each repeatedly running one of the
     K inputs in a randomized order.
  3. Assert every concurrent output bit-for-bit equals its reference.

Any data race inside the shared session (e.g. clobbered intermediate buffers)
would surface as a mismatch. Identical outputs across thousands of interleaved
runs is strong evidence the session may be shared safely.
"""

from __future__ import annotations

import argparse
import threading

import common
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--reps", type=int, default=200, help="runs per thread")
    parser.add_argument("--inputs", type=int, default=4, help="distinct fixed inputs")
    args = parser.parse_args()

    model_path = common.resolve_model_path(args.model)
    common.print_environment(model_path)

    session = common.make_session(model_path, intra_op=1)

    # Fixed inputs + single-threaded reference outputs.
    inputs = [common.make_input(seed=i) for i in range(args.inputs)]
    refs = [np.array(session.run(None, {common.INPUT_NAME: x})[0]) for x in inputs]

    mismatches: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(args.threads)

    def worker(tid: int) -> None:
        rng = np.random.default_rng(1000 + tid)
        barrier.wait()
        for _ in range(args.reps):
            idx = int(rng.integers(0, args.inputs))
            out = np.asarray(session.run(None, {common.INPUT_NAME: inputs[idx]})[0])
            if not np.array_equal(out, refs[idx]):
                with lock:
                    mismatches.append(
                        f"thread {tid}: input {idx} got {out.ravel()[:3]} "
                        f"expected {refs[idx].ravel()[:3]}"
                    )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(args.threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = args.threads * args.reps
    print(
        f"\nConcurrent runs : {total} "
        f"({args.threads} threads x {args.reps} reps over {args.inputs} inputs)"
    )
    if mismatches:
        print(f"RESULT: FAIL - {len(mismatches)} mismatched outputs")
        for m in mismatches[:10]:
            print("  " + m)
        raise SystemExit(1)
    print(
        "RESULT: PASS - every concurrent output bit-for-bit matched its single-threaded reference."
    )
    print("=> Concurrent Run() on a single shared session is correct (CPU EP).")


if __name__ == "__main__":
    main()
