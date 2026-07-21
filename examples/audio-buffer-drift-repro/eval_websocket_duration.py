#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio eval: recordings over the WebSocket path must match the call timeline.

Runs the scenarios from ``websocket_repro.py`` (real Twilio media messages
through the real ``TwilioFrameSerializer`` into a real ``AudioBufferProcessor``)
and asserts that the recorded track duration stays within a small tolerance of
the real call timeline. Reported in the pass/fail style of ``pipecat eval run``.

Note on framing: ``pipecat eval`` scenarios (``pipecat.evals``) assert on RTVI
conversation events from a live bot over WebSocket; they have no expectation
type for recorded-audio duration, and the scripted message timing here has to
be driven from inside the process. So this eval drives the same in-process
pipeline as ``websocket_repro.py`` and applies the duration assertion directly,
matching the approach of ``eval_duration_drift.py``.

Both scenarios pass with the capture-timestamp (pts) positioning fix. Checked
out without it (the previous commit on this branch), the mute-gap scenario
FAILS: the burst arrival is read as a stall catch-up and about 2 s of the
genuine 3 s mute gap gets trimmed (issue #4561 again), so the recording comes
out around 5 s instead of 7 s.

Run:
    cd <pipecat repo root>
    .venv/bin/python examples/audio-buffer-drift-repro/eval_websocket_duration.py
"""

import asyncio
import sys

from loguru import logger
from websocket_repro import mute_gap_burst_events, network_stall_burst_events, run_scenario

# Recorded duration may legitimately differ from the timeline by scheduler
# jitter and the stream resampler's warm-up loss (the silence-gap threshold
# itself is 200 ms). Anything beyond this is a positioning error.
TOLERANCE_SECS = 0.25

SCENARIOS = [
    ("websocket_recording_duration_mute_gap_burst", mute_gap_burst_events),
    ("websocket_recording_duration_network_stall_burst", network_stall_burst_events),
]


async def main() -> int:
    failed = 0
    for name, build in SCENARIOS:
        events, timeline_secs = build()
        r = await run_scenario(events=events, timeline_secs=timeline_secs)
        ok = abs(r["drift_secs"]) <= TOLERANCE_SECS
        verdict = "PASSED" if ok else "FAILED"
        print(
            f"[{verdict}] {name}: real audio {r['fed_secs']:.3f}s, "
            f"timeline {r['timeline_secs']:.3f}s, recorded {r['recorded_secs']:.3f}s, "
            f"drift {r['drift_secs']:+.3f}s (tolerance ±{TOLERANCE_SECS:.2f}s)"
        )
        if not ok:
            failed += 1
            print(
                "         recording diverges from the call timeline: a genuine "
                "mute gap was trimmed as if it were a stall catch-up burst "
                "(capture timestamps not honored)"
            )
    print(f"\n{len(SCENARIOS) - failed}/{len(SCENARIOS)} scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    sys.exit(asyncio.run(main()))
