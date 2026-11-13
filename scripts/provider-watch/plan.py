#!/usr/bin/env python3
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Slice the provider-watch unit list into groups for the workflow's matrix.

Each group is small enough for one research job to finish inside its GitHub App
token's one-hour lifetime. Emits a JSON array of ``{"name", "units"}`` objects:
``name`` labels the matrix job and its artifact, ``units`` is the comma-joined
``--only`` value the job passes to the skill. Stdlib only, so the plan job runs
it without installing anything. Run::

    python3 scripts/provider-watch/plan.py --group-size 12 [--only ...] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inventory  # noqa: E402


def plan_groups(unit_ids: list[str], group_size: int) -> list[dict[str, str]]:
    """Chunk unit ids, in order, into named groups of at most ``group_size``."""
    groups = []
    for start in range(0, len(unit_ids), group_size):
        chunk = unit_ids[start : start + group_size]
        first = chunk[0].split("/")[0]
        last = chunk[-1].split("/")[0]
        span = first if first == last else f"{first}-{last}"
        groups.append({"name": f"g{start // group_size + 1:02d}-{span}", "units": ",".join(chunk)})
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="emit JSON (default)")
    parser.add_argument("--group-size", type=int, default=12, help="max units per group")
    parser.add_argument(
        "--only", help="comma-separated providers or unit ids (e.g. openai,deepgram/stt)"
    )
    parser.add_argument("--limit", type=int, help="keep only the first N units")
    args = parser.parse_args(argv)

    units = inventory.scan_services()
    units = inventory.select(units, args.only.split(",") if args.only else None, args.limit)
    print(json.dumps(plan_groups([u.id for u in units], args.group_size)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
