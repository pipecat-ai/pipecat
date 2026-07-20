#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Run the stall+burst scenario and save the recorded track as a WAV file.

Companion to ``eval_duration_drift.py``: that script asserts on durations,
this one produces an audio file you can actually listen to / inspect in an
editor, so a reviewer can hear the difference rather than just read numbers.

Run:
    cd <pipecat repo root>
    .venv/bin/python examples/audio-buffer-drift-repro/dump_wav.py <output.wav>
"""

import asyncio
import sys
import wave

from loguru import logger
from repro import BYTES_PER_SECOND, SAMPLE_RATE, run_scenario_with_audio


async def main(out_path: str):
    result = await run_scenario_with_audio(total_secs=10.0, stalls={150: 0.6, 300: 0.6})
    audio = result["audio"]
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio)
    print(
        f"wrote {out_path}: real audio {result['fed_secs']:.3f}s, "
        f"recorded {len(audio) / BYTES_PER_SECOND:.3f}s, "
        f"drift {result['drift_secs']:+.3f}s"
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    out = sys.argv[1] if len(sys.argv) > 1 else "stall_burst.wav"
    asyncio.run(main(out))
