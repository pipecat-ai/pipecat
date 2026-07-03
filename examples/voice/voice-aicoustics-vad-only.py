#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Minimal VAD-only test bot for AICQuailVADAnalyzer.

Strips out STT / LLM / TTS / audio recording so the only things exercised on
the audio path are the AICFilter (noise cancellation) and the
AICQuailVADAnalyzer (voice activity detection). Useful for sanity-checking
VAD behavior under speech / silence / background noise without paying for
STT/LLM/TTS API calls.

Wiring note: in modern Pipecat, VADAnalyzer instances are driven by a
:class:`pipecat.processors.audio.vad_processor.VADProcessor` placed in the
pipeline (not by ``TransportParams.vad_analyzer``). This example puts the
``VADProcessor`` right after ``transport.input()`` so raw VAD events
(``VADUserStartedSpeakingFrame`` / ``VADUserStoppedSpeakingFrame``) reach
the logger without going through a turn-strategy layer.

Logging:
    - INFO ">>> VAD: user started speaking" / "<<< VAD: user stopped speaking"
      for the raw VAD events emitted by the VADProcessor.
    - INFO "VAD heartbeat" every ~1s with: current state, total frames analyzed
      in the window, how many crossed the speech-confidence threshold, and the
      most recent raw confidence value. Speech% rising as you talk and falling
      to ~0% in silence is the direct signal that the VAD is processing audio
      correctly.
    - INFO "audio frame batch" every ~5s confirming input audio is flowing.
    - DEBUG init lines from AICFilter + AICQuailVADAnalyzer (run with
      LOGURU_LEVEL=DEBUG to see them).

Required env vars:
    AIC_SDK_LICENSE    ai-coustics SDK license key
    Plus whatever credentials the chosen transport needs (DAILY_*, etc.)

Run:
    LOGURU_LEVEL=DEBUG uv run python examples/voice/voice-aicoustics-vad-only.py daily
"""

import datetime
import os
import time

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.filters.aic_filter import AICFilter
from pipecat.audio.vad.aic_quail_vad import AICQuailVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


# Roughly one heartbeat per second at the analyzer's 10ms-window cadence.
_HEARTBEAT_FRAMES = 100
# Audio-flow heartbeat: every ~5 seconds at ~50 audio frames/sec.
_AUDIO_HEARTBEAT_FRAMES = 250


class LoggingAICQuailVADAnalyzer(AICQuailVADAnalyzer):
    """Adds a once-per-second heartbeat with confidence + state stats."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lv_call_count = 0
        self._lv_speech_count = 0
        self._lv_window_start = time.monotonic()

    def voice_confidence(self, buffer: bytes) -> float:
        confidence = super().voice_confidence(buffer)
        self._lv_call_count += 1
        if confidence >= 0.5:
            self._lv_speech_count += 1

        if self._lv_call_count >= _HEARTBEAT_FRAMES:
            now = time.monotonic()
            elapsed = now - self._lv_window_start
            pct = (self._lv_speech_count / self._lv_call_count) * 100
            state_name = getattr(getattr(self, "_vad_state", None), "name", "?")
            logger.info(
                f"VAD heartbeat: state={state_name} "
                f"frames={self._lv_call_count} in {elapsed:.2f}s "
                f"speech={self._lv_speech_count} ({pct:.0f}%) "
                f"last_conf={confidence:.1f}"
            )
            self._lv_call_count = 0
            self._lv_speech_count = 0
            self._lv_window_start = now

        return confidence


class VADEventLogger(FrameProcessor):
    """Print VAD turn boundaries + an occasional audio-flow heartbeat."""

    def __init__(self) -> None:
        super().__init__()
        self._audio_frames_total = 0
        self._audio_frames_since_log = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if isinstance(frame, VADUserStartedSpeakingFrame):
            logger.info(f"[{ts}] >>> VAD: user started speaking")
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            logger.info(f"[{ts}] <<< VAD: user stopped speaking")
        elif isinstance(frame, InputAudioRawFrame):
            self._audio_frames_total += 1
            self._audio_frames_since_log += 1
            if self._audio_frames_since_log >= _AUDIO_HEARTBEAT_FRAMES:
                logger.info(
                    f"[{ts}] audio frame batch: +{self._audio_frames_since_log} "
                    f"(total {self._audio_frames_total})"
                )
                self._audio_frames_since_log = 0
        elif isinstance(frame, StartFrame):
            logger.info(f"[{ts}] pipeline StartFrame")
        elif isinstance(frame, EndFrame):
            logger.info(f"[{ts}] pipeline EndFrame")

        await self.push_frame(frame, direction)


aic_filter = AICFilter(
    license_key=os.environ["AIC_SDK_LICENSE"],
    model_id="quail-vf-2.2-l-16khz",
    enhancement_level=0.8,
)
aic_vad_analyzer = LoggingAICQuailVADAnalyzer(
    license_key=os.environ["AIC_SDK_LICENSE"],
)

# VAD is driven by a VADProcessor in the pipeline (modern pipecat path).
# TransportParams here only enables audio input + the AICFilter; the VAD
# analyzer is wired via VADProcessor below.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_in_filter=aic_filter,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_in_filter=aic_filter,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_in_filter=aic_filter,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    logger.info("VAD-only test bot starting")
    vad_processor = VADProcessor(vad_analyzer=aic_vad_analyzer)
    pipeline = Pipeline([transport.input(), vad_processor, VADEventLogger()])
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=False))
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
