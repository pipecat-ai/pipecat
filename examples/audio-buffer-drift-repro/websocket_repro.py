#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Reproduction: over WebSocket, the stall+burst trim can eat real mute gaps.

The stall+burst reconciliation in ``AudioBufferProcessor`` (see ``repro.py``)
assumes incoming audio is paced to real time, so a frame that would push the
buffer past the wall-clock position must be late audio from a stall. That
holds for WebRTC, which jitter-buffers. It does NOT hold for WebSocket
transports: audio arrives at whatever rate the network and the client
produce. A genuine gap (muted mic) followed by audio that happens to arrive
in a burst looks exactly like a stall followed by a catch-up burst, and the
reconciliation trims the silence that legitimately represents the mute gap.
That shortens the recording and pushes the two utterances back together,
which is issue #4561 all over again.

The fix (Filipi's review on PR #5071): position audio by a capture timestamp
from the client instead of inferring it from arrival pacing. Twilio media
messages carry a ``timestamp`` field (ms since stream start). The serializer
stamps it onto the frame's ``pts`` and the processor places the audio at that
capture time. Transports without a timestamp keep the arrival-pacing
behavior.

This script drives the REAL ``TwilioFrameSerializer.deserialize`` with
scripted Twilio-format ``media`` JSON messages (real base64 u-law payloads
and real ``timestamp`` values) and feeds the resulting frames into a real
``AudioBufferProcessor`` inside a ``Pipeline``, no live phone call needed.

Scenarios:

- ``mute gap + burst arrival``: 2 s of audio, a genuine 3 s mute gap (no
  media messages), then 2 s of audio whose messages arrive back-to-back.
  The real timeline is 7 s. Without capture-time positioning the trim eats
  most of the mute-gap silence and the recording comes out around 5 s.
- ``network stall + late burst``: 2 s of audio where delivery stalls for
  0.6 s mid-stream and the backlog then arrives in a burst. Capture
  timestamps stay contiguous, so the recording must stay at 2 s (the
  over-count fixed by PR #5071 must not come back).

Run:
    cd <pipecat repo root>
    .venv/bin/python examples/audio-buffer-drift-repro/websocket_repro.py
"""

import asyncio
import base64
import json
import time

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.workers.runner import WorkerRunner

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16-bit mono at the pipeline rate
CHUNK_MS = 20
ULAW_CHUNK_BYTES = 8000 * CHUNK_MS // 1000  # 8 kHz u-law, 1 byte per sample

# A non-silent u-law payload (0x10 decodes to a large PCM value), so real
# audio is distinguishable from injected silence.
MEDIA_PAYLOAD = base64.b64encode(b"\x10" * ULAW_CHUNK_BYTES).decode("utf-8")


def media_message(timestamp_ms: int) -> str:
    """Build one Twilio Media Streams ``media`` message.

    Matches the wire format documented at
    https://www.twilio.com/docs/voice/media-streams/websocket-messages
    (``media.timestamp`` is the capture time in ms since stream start,
    sent as a string).
    """
    return json.dumps(
        {
            "event": "media",
            "sequenceNumber": str(timestamp_ms // CHUNK_MS + 1),
            "media": {
                "track": "inbound",
                "chunk": str(timestamp_ms // CHUNK_MS + 1),
                "timestamp": str(timestamp_ms),
                "payload": MEDIA_PAYLOAD,
            },
            "streamSid": "MZ0000000000000000000000000000000000",
        }
    )


class SimulatedTwilioWebsocket(FrameProcessor):
    """Feeds scripted Twilio media messages through the real serializer.

    Each event is ``(send_at_secs, message)``: the message is deserialized
    with ``TwilioFrameSerializer.deserialize`` and pushed at that wall-clock
    offset. Events sharing a ``send_at_secs`` are delivered back-to-back,
    which is how a WebSocket delivers a burst.

    Args:
        events: Scripted ``(send_at_secs, twilio_json_message)`` sequence.
    """

    def __init__(self, *, events: list[tuple[float, str]], **kwargs):
        super().__init__(**kwargs)
        self._events = events
        self._serializer = TwilioFrameSerializer(
            "MZ0000000000000000000000000000000000",
            params=TwilioFrameSerializer.InputParams(auto_hang_up=False),
        )
        self._ws_task = None
        self.done = asyncio.Event()
        self.audio_bytes_fed = 0
        self.wall_secs = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame) and self._ws_task is None:
            await self._serializer.setup(frame)
            self._ws_task = self.create_task(self._run_ws())
        elif isinstance(frame, (EndFrame, CancelFrame)) and self._ws_task is not None:
            await self.cancel_task(self._ws_task)
            self._ws_task = None
        await self.push_frame(frame, direction)

    async def _run_ws(self):
        loop = asyncio.get_running_loop()
        start = time.monotonic()
        loop_start = loop.time()
        for send_at, message in self._events:
            delay = loop_start + send_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            # else: this message is part of a burst -> deliver immediately.
            frame = await self._serializer.deserialize(message)
            if frame is not None:
                self.audio_bytes_fed += len(frame.audio)
                await self.push_frame(frame)
        self.wall_secs = time.monotonic() - start
        self.done.set()


def mute_gap_burst_events() -> tuple[list[tuple[float, str]], float]:
    """2 s audio, genuine 3 s mute gap, then 2 s audio arriving as a burst.

    Capture timestamps jump across the mute gap (Twilio stops sending while
    there is no audio), so the second utterance is stamped 5000..6980 ms.
    Its messages all arrive at wall-clock 5.0 s, back-to-back, the way a
    client that buffered briefly flushes them. Real timeline: 7 s.
    """
    events: list[tuple[float, str]] = []
    for i in range(100):  # utterance A, paced in real time
        events.append((i * 0.02, media_message(i * CHUNK_MS)))
    for i in range(100):  # utterance B, burst arrival after the mute gap
        events.append((5.0, media_message(5000 + i * CHUNK_MS)))
    return events, 7.0


def network_stall_burst_events() -> tuple[list[tuple[float, str]], float]:
    """2 s of continuous audio with a 0.6 s delivery stall mid-stream.

    Capture timestamps are contiguous (the audio existed on the wire, it was
    just delivered late), and the backlog arrives as a burst once delivery
    resumes. Real timeline: 2 s. This is the over-count case PR #5071 fixes.
    """
    events: list[tuple[float, str]] = []
    for i in range(100):
        send_at = i * 0.02
        if i >= 50:
            # Delivery stalls at 1.0 s for 0.6 s; the backlog (frames 50-79)
            # then arrives back-to-back before pacing resumes.
            send_at = max(send_at, 1.6)
        events.append((send_at, media_message(i * CHUNK_MS)))
    return events, 2.0


async def run_scenario_with_audio(*, events: list[tuple[float, str]], timeline_secs: float) -> dict:
    """Run one recording scenario and return measured durations plus the raw recorded audio."""
    ws = SimulatedTwilioWebsocket(events=events)
    audiobuffer = AudioBufferProcessor(
        sample_rate=SAMPLE_RATE, num_channels=1, auto_start_recording=True
    )

    captured: dict = {}

    @audiobuffer.event_handler("on_track_audio_data")
    async def on_track_audio_data(buffer, user_audio, bot_audio, sample_rate, num_channels):
        captured["user"] = user_audio
        captured["bot"] = bot_audio

    # In a real bot the AudioBufferProcessor sits after transport.output(); here
    # the simulated websocket replaces the transport so the run needs no services.
    pipeline = Pipeline([ws, audiobuffer])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE),
        cancel_on_idle_timeout=False,
    )

    async def driver():
        await ws.done.wait()
        await asyncio.sleep(0.2)  # let any queued frames drain into the processor
        await audiobuffer.stop_recording()
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await asyncio.gather(runner.run(), driver())

    audio = captured.get("user", b"")
    fed_secs = ws.audio_bytes_fed / BYTES_PER_SECOND
    recorded_secs = len(audio) / BYTES_PER_SECOND
    return {
        "audio": audio,
        "fed_secs": fed_secs,
        "timeline_secs": timeline_secs,
        "recorded_secs": recorded_secs,
        "wall_secs": ws.wall_secs,
        "drift_secs": recorded_secs - timeline_secs,
    }


async def run_scenario(*, events: list[tuple[float, str]], timeline_secs: float) -> dict:
    """Run one recording scenario and return the measured durations (seconds)."""
    result = await run_scenario_with_audio(events=events, timeline_secs=timeline_secs)
    return {k: v for k, v in result.items() if k != "audio"}


SCENARIOS = [
    ("mute gap + burst arrival", mute_gap_burst_events),
    ("network stall + late burst", network_stall_burst_events),
]


async def main():
    print()
    print(f"{'scenario':<30} {'real audio':>10} {'timeline':>9} {'recorded':>9} {'drift':>8}")
    print("-" * 72)
    for name, build in SCENARIOS:
        events, timeline_secs = build()
        r = await run_scenario(events=events, timeline_secs=timeline_secs)
        print(
            f"{name:<30} {r['fed_secs']:>9.3f}s {r['timeline_secs']:>8.3f}s "
            f"{r['recorded_secs']:>8.3f}s {r['drift_secs']:>+7.3f}s"
        )
    print()
    print(
        "The recorded track should match the real timeline in both scenarios.\n"
        "Without capture-time (pts) positioning, the mute-gap scenario comes out\n"
        "SHORT: the burst arrival is read as a stall catch-up and the mute-gap\n"
        "silence is trimmed away (issue #4561 again). With pts positioning both\n"
        "scenarios match the timeline."
    )


if __name__ == "__main__":
    import sys

    # Keep the output readable: pipeline INFO logs go to stderr at WARNING+.
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    asyncio.run(main())
