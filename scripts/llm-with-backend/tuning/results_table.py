#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Print a suite run's results as a Markdown table, for the tuning RESULTS.md.

Each row is a scenario and each column a bot, with how many of the runs
passed::

    uv run python scripts/llm-with-backend/tuning/results_table.py test-runs/<run>/results.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Print a suite run's results as a Markdown table.")
    parser.add_argument("results", type=Path, help="The run's results.jsonl.")
    args = parser.parse_args()

    passed: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for line in args.results.open():
        result = json.loads(line)
        bot = result["bot"].removesuffix(".py")
        passed[(result["scenario"], bot)].append(result["passed"])

    bots = sorted({bot for _, bot in passed})
    scenarios = sorted({scenario for scenario, _ in passed})
    print("| scenario | " + " | ".join(bots) + " |")
    print("| --- |" + " --- |" * len(bots))
    for scenario in scenarios:
        cells = []
        for bot in bots:
            runs = passed.get((scenario, bot), [])
            cells.append(f"{sum(runs)}/{len(runs)}" if runs else "")
        print(f"| `{scenario}` | " + " | ".join(cells) + " |")
    totals = []
    for bot in bots:
        runs = [p for (_, b), results in passed.items() if b == bot for p in results]
        totals.append(f"**{sum(runs)}/{len(runs)}**")
    print("| **total** | " + " | ".join(totals) + " |")


if __name__ == "__main__":
    main()
