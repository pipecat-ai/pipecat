#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Compare two runs of the turn-completion eval, model by model and kind by kind.

Usage::

    uv run python scripts/turn-completion-evals/compare.py runs/baseline runs/prompt-v2

Only models and steps present in both runs are compared, so a run that was
stopped early or excluded providers still lines up against a full one.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(run_dir: Path) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(run_dir.glob("*.jsonl")):
        for line in path.open():
            if line.strip():
                r = json.loads(line)
                by_model[r["model"]].append(r)
    return by_model


def rate(rows: list[dict], key: str = "passed") -> float:
    return 100 * sum(bool(r[key]) for r in rows) / len(rows) if rows else float("nan")


def main(a_dir: str, b_dir: str) -> int:
    a, b = load(Path(a_dir)), load(Path(b_dir))
    models = sorted(set(a) & set(b))
    if not models:
        print("no models in common")
        return 1

    # Step identity: (case, step, attempt). Keep only steps scored in both runs
    # (an errored step in either is dropped, since it says nothing about the prompt).
    def keyed(rows):
        return {(r["case"], r["step"], r["attempt"]): r for r in rows if not r["error"]}

    print(
        f"{'model':46} {'A pass':>7} {'B pass':>7} {'delta':>6}  {'A verd':>6} {'B verd':>6}  {'A fmt':>6} {'B fmt':>6}  n"
    )
    tot_a: list[dict] = []
    tot_b: list[dict] = []
    deltas = []
    for m in models:
        ka, kb = keyed(a[m]), keyed(b[m])
        common = sorted(set(ka) & set(kb))
        ra, rb = [ka[k] for k in common], [kb[k] for k in common]
        if not common:
            continue
        tot_a += ra
        tot_b += rb
        d = rate(rb) - rate(ra)
        deltas.append((d, m, ra, rb))
    for d, m, ra, rb in sorted(deltas, key=lambda x: x[0]):
        print(
            f"{m:46} {rate(ra):6.1f}% {rate(rb):6.1f}% {d:+6.1f}  "
            f"{rate(ra, 'verdict_ok'):5.1f}% {rate(rb, 'verdict_ok'):5.1f}%  "
            f"{rate(ra, 'format_ok'):5.1f}% {rate(rb, 'format_ok'):5.1f}%  {len(ra)}"
        )
    print(
        f"\n{'ALL':46} {rate(tot_a):6.1f}% {rate(tot_b):6.1f}% {rate(tot_b) - rate(tot_a):+6.1f}  "
        f"{rate(tot_a, 'verdict_ok'):5.1f}% {rate(tot_b, 'verdict_ok'):5.1f}%  "
        f"{rate(tot_a, 'format_ok'):5.1f}% {rate(tot_b, 'format_ok'):5.1f}%  {len(tot_a)}"
    )

    print("\nFailure kinds (A -> B):")
    ka, kb = (
        Counter(k for r in tot_a for k in r["kinds"]),
        Counter(k for r in tot_b for k in r["kinds"]),
    )
    for kind in sorted(set(ka) | set(kb), key=lambda k: -(ka[k] + kb[k])):
        print(f"  {kind:24} {ka[kind]:5} -> {kb[kind]:5}  ({kb[kind] - ka[kind]:+d})")

    print("\nBy category (A -> B):")
    ca, cb = defaultdict(list), defaultdict(list)
    for r in tot_a:
        ca[r["category"]].append(r)
    for r in tot_b:
        cb[r["category"]].append(r)
    for c in sorted(ca, key=lambda c: rate(cb[c]) - rate(ca[c])):
        print(
            f"  {c:16} {rate(ca[c]):5.1f}% -> {rate(cb[c]):5.1f}%  ({rate(cb[c]) - rate(ca[c]):+.1f})"
        )

    print("\nSteps that moved most (fails across models, A -> B):")
    sa, sb = defaultdict(list), defaultdict(list)
    for r in tot_a:
        sa[(r["case"], r["step"])].append(r)
    for r in tot_b:
        sb[(r["case"], r["step"])].append(r)
    moved = []
    for k in sa:
        fa = sum(not r["passed"] for r in sa[k])
        fb = sum(not r["passed"] for r in sb[k])
        moved.append((fb - fa, fa, fb, k))
    moved.sort()
    for d, fa, fb, (case, step) in moved[:10] + [x for x in moved[-10:] if x[0] > 0]:
        print(f"  {case:44} step {step}  {fa:3} -> {fb:3}  ({d:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
