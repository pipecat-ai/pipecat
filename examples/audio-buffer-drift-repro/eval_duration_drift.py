#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio eval: AudioBufferProcessor recordings must match real audio duration.

Runs the scenarios from ``repro.py`` and asserts that the recorded track
duration stays within a small tolerance of the real audio fed in. Reported in
the pass/fail style of ``pipecat eval run``.

Note on framing: ``pipecat eval`` scenarios (``pipecat.evals``) assert on RTVI
conversation events from a live bot over WebSocket; they have no expectation
type for recorded-audio duration, and a deterministic event-loop stall has to
be induced inside the bot process. So this eval drives the same in-process
pipeline as ``repro.py`` and applies the duration assertion directly.

This PR fixes the double-count in ``_fill_buffer_silence_gap``
(``src/pipecat/processors/audio/audio_buffer_processor.py``), so both
scenarios pass against the code in this branch. Checked out before this fix
(e.g. main @ 5b5cc30ff), the stall scenario FAILS and this script exits
non-zero; see README.md for the before/after numbers and WAV files.

Run:
    cd <pipecat repo root>
    .venv/bin/python examples/audio-buffer-drift-repro/eval_duration_drift.py
"""

import asyncio
import sys

from loguru import logger
from repro import run_scenario

# Recorded duration may legitimately differ from fed duration by scheduler
# jitter (the silence-gap threshold itself is 200 ms). Anything beyond this is
# phantom audio.
TOLERANCE_SECS = 0.25

SCENARIOS = [
    ("recording_duration_steady", {"total_secs": 10.0, "stalls": None}),
    (
        "recording_duration_stall_burst",
        {"total_secs": 10.0, "stalls": {150: 0.6, 300: 0.6}},
    ),
]


async def main() -> int:
    failed = 0
    for name, kwargs in SCENARIOS:
        r = await run_scenario(**kwargs)
        ok = abs(r["drift_secs"]) <= TOLERANCE_SECS
        verdict = "PASSED" if ok else "FAILED"
        print(
            f"[{verdict}] {name}: real audio {r['fed_secs']:.3f}s, "
            f"wall clock {r['wall_secs']:.3f}s, recorded {r['recorded_secs']:.3f}s, "
            f"drift {r['drift_secs']:+.3f}s (tolerance ±{TOLERANCE_SECS:.2f}s)"
        )
        if not ok:
            failed += 1
            print(
                f"         recording overshoots real audio by {r['drift_secs']:.3f}s "
                f"(total scripted stall time: {r['total_stall_secs']:.3f}s): "
                f"stall windows are double-counted as silence + late real audio"
            )
    print(f"\n{len(SCENARIOS) - failed}/{len(SCENARIOS)} scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    sys.exit(asyncio.run(main()))
