#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Reproduction: AudioBufferProcessor recordings grow past real call length.

`AudioBufferProcessor._fill_buffer_silence_gap` injects silence when more than
200 ms of wall-clock time passes between writes to a track buffer. That is the
right call when the gap means "no audio existed on the wire" (muted mic, idle
bot). But when the gap means "audio existed but was delivered late" (an event
loop stall followed by a catch-up burst of queued frames), the correction
double-counts: the stall window is added once as injected silence and once as
the real audio that arrives in the burst right after. There is no negative-gap
or trim handling, so the recording ends up LONGER than the real call. See
`_fill_buffer_silence_gap` in
``src/pipecat/processors/audio/audio_buffer_processor.py``.

This script demonstrates the bug with a real ``Pipeline`` run by a
``PipelineWorker``, no mocks and no external services:

- ``SimulatedUserMic`` stands in for ``transport.input()``. It pushes one 20 ms
  ``InputAudioRawFrame`` per tick on a fixed real-time schedule and, like a real
  transport draining its socket backlog, bursts frames back-to-back after it
  falls behind.
- At scripted points it calls ``time.sleep()`` to BLOCK the event loop, exactly
  what a slow synchronous callback, GC pause, or blocked thread does to a
  production bot.
- ``AudioBufferProcessor`` sits downstream, as it would after
  ``transport.output()`` in a real bot. When recording stops, we compare the
  duration of real audio fed in against the duration of the recorded track.

Expected (buggy) result: every stall adds roughly its own duration of phantom
audio to the recording. Two 600 ms stalls in a 10 s call produce a recording of
about 11.2 s.

Run:
    cd <pipecat repo root>
    .venv/bin/python examples/audio-buffer-drift-repro/repro.py
"""

import asyncio
import time

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

SAMPLE_RATE = 16000
CHUNK_MS = 20
BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16-bit mono


class SimulatedUserMic(FrameProcessor):
    """Pushes 20 ms user audio frames at real-time cadence, with scripted stalls.

    Modeled on the catch-up scheduling of ``pipecat.evals.transport.EvalMicrophone``:
    frames are scheduled on a fixed timeline (``next_send += tick``), and when the
    loop wakes up late the backlog is sent back-to-back until the schedule is
    caught up. That catch-up burst is the exact delivery pattern a real transport
    produces after an event-loop stall, and it is what triggers the bug.

    Args:
        total_secs: Total seconds of real audio to feed.
        stalls: Mapping of frame index -> seconds to BLOCK the event loop
            (via ``time.sleep``) right before sending that frame.
    """

    def __init__(self, *, total_secs: float, stalls: dict[int, float] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._num_frames = int(total_secs * 1000 / CHUNK_MS)
        self._stalls = stalls or {}
        self._mic_task = None
        self.done = asyncio.Event()
        self.bytes_fed = 0
        self.wall_secs = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame) and self._mic_task is None:
            self._mic_task = self.create_task(self._run_mic())
        elif isinstance(frame, (EndFrame, CancelFrame)) and self._mic_task is not None:
            await self.cancel_task(self._mic_task)
            self._mic_task = None
        await self.push_frame(frame, direction)

    async def _run_mic(self):
        chunk_samples = SAMPLE_RATE * CHUNK_MS // 1000
        # Non-zero PCM, so real audio is distinguishable from injected silence.
        chunk = b"\x10\x00" * chunk_samples
        tick = CHUNK_MS / 1000
        loop = asyncio.get_running_loop()
        start = time.monotonic()
        next_send = loop.time()
        for i in range(self._num_frames):
            stall = self._stalls.get(i)
            if stall:
                # Deliberately BLOCK the event loop (a slow sync callback, a GC
                # pause, a blocked thread...). Frames that "should" have been
                # sent during this window burst out right after, back-to-back.
                time.sleep(stall)
            await self.push_frame(
                InputAudioRawFrame(audio=chunk, sample_rate=SAMPLE_RATE, num_channels=1)
            )
            self.bytes_fed += len(chunk)
            next_send += tick
            now = loop.time()
            if next_send > now:
                await asyncio.sleep(next_send - now)
            # else: behind schedule -> loop immediately (catch-up burst).
        self.wall_secs = time.monotonic() - start
        self.done.set()


async def run_scenario_with_audio(
    *, total_secs: float, stalls: dict[int, float] | None = None
) -> dict:
    """Run one recording scenario and return measured durations plus the raw recorded audio."""
    mic = SimulatedUserMic(total_secs=total_secs, stalls=stalls)
    audiobuffer = AudioBufferProcessor(
        sample_rate=SAMPLE_RATE, num_channels=1, auto_start_recording=True
    )

    captured: dict = {}

    @audiobuffer.event_handler("on_track_audio_data")
    async def on_track_audio_data(buffer, user_audio, bot_audio, sample_rate, num_channels):
        captured["user"] = user_audio
        captured["bot"] = bot_audio

    # In a real bot the AudioBufferProcessor sits after transport.output(); here
    # the simulated mic replaces the transport so the run needs no services.
    pipeline = Pipeline([mic, audiobuffer])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE),
        cancel_on_idle_timeout=False,
    )

    async def driver():
        await mic.done.wait()
        await asyncio.sleep(0.2)  # let any queued frames drain into the processor
        await audiobuffer.stop_recording()
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await asyncio.gather(runner.run(), driver())

    audio = captured.get("user", b"")
    fed_secs = mic.bytes_fed / BYTES_PER_SECOND
    recorded_secs = len(audio) / BYTES_PER_SECOND
    return {
        "audio": audio,
        "fed_secs": fed_secs,
        "recorded_secs": recorded_secs,
        "wall_secs": mic.wall_secs,
        "drift_secs": recorded_secs - fed_secs,
        "total_stall_secs": sum((stalls or {}).values()),
    }


async def run_scenario(*, total_secs: float, stalls: dict[int, float] | None = None) -> dict:
    """Run one recording scenario and return the measured durations (seconds)."""
    result = await run_scenario_with_audio(total_secs=total_secs, stalls=stalls)
    return {k: v for k, v in result.items() if k != "audio"}


async def main():
    scenarios = [
        ("steady (no stalls)", {"total_secs": 10.0, "stalls": None}),
        (
            "2 x 600ms stall + catch-up burst",
            {"total_secs": 10.0, "stalls": {150: 0.6, 300: 0.6}},
        ),
    ]

    print()
    print(f"{'scenario':<36} {'real audio':>10} {'wall clock':>10} {'recorded':>10} {'drift':>8}")
    print("-" * 80)
    for name, kwargs in scenarios:
        r = await run_scenario(**kwargs)
        print(
            f"{name:<36} {r['fed_secs']:>9.3f}s {r['wall_secs']:>9.3f}s "
            f"{r['recorded_secs']:>9.3f}s {r['drift_secs']:>+7.3f}s"
        )
    print()
    print(
        "The recorded track should match the real audio fed in, including across\n"
        "stalls. On the unfixed code, the stall run instead comes out longer by\n"
        "roughly the total stall time, because the stall window gets counted\n"
        "twice: once as injected silence and once as the late-delivered real\n"
        "audio. See dump_wav.py to capture the recorded audio and README.md for\n"
        "before/after WAV files."
    )


if __name__ == "__main__":
    import sys

    # Keep the output readable: pipeline INFO logs go to stderr at WARNING+.
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    asyncio.run(main())
