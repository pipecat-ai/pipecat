#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""voice-agent-freeze — Pipecat voice agent with optional freeze simulation.

Cascade: Speech-to-Text → LLM → Text-to-Speech.

Run::

    uv run app.py
"""

from __future__ import annotations

import io
import json
import os
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from loguru import logger
from dotenv import load_dotenv
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from freeze_simulator import FreezeSimulator
from turn_manager import TurnManager

load_dotenv(override=True)

RECORDINGS_DIR = Path("recordings")
TRANSCRIPTS_DIR = Path("transcripts")

_DEFAULT_VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"
_DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash"
_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, "
    "so avoid emojis, bullet points, or other formatting that can't be spoken. "
    "Respond to what the user said in a creative, helpful, and brief way."
)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value or not value.strip():
        msg = f"Missing or empty required environment variable: {name}"
        raise RuntimeError(msg)
    return value


async def save_audio_file(
    audio: bytes,
    path: Path,
    sample_rate: int,
    num_channels: int,
) -> None:
    """Save PCM audio to a WAV file."""
    if not audio:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wf:
            wf.setsampwidth(2)
            wf.setnchannels(num_channels)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)
        async with aiofiles.open(path, "wb") as f:
            await f.write(buffer.getvalue())
    logger.info("Audio saved to {}", path)


async def run_bot(transport: BaseTransport) -> None:
    """Wire STT, LLM, TTS, freeze simulator, and session export handlers."""
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=_require_env("DEEPGRAM_API_KEY"))
    tts = ElevenLabsTTSService(
        api_key=_require_env("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(
            voice=os.environ.get("ELEVENLABS_VOICE_ID", _DEFAULT_VOICE_ID),
        ),
    )
    google_model = os.environ.get("GOOGLE_MODEL", "").strip() or _DEFAULT_GOOGLE_MODEL
    llm = GoogleLLMService(
        api_key=_require_env("GOOGLE_API_KEY"),
        settings=GoogleLLMService.Settings(
            model=google_model,
            system_instruction=_SYSTEM_INSTRUCTION,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    audio_buffer = AudioBufferProcessor()
    turn_manager = TurnManager()
    latency_observer = UserBotLatencyObserver()
    freeze_simulator = FreezeSimulator(freeze_after_turn=2)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            freeze_simulator,
            transport.output(),
            audio_buffer,
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[latency_observer],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client) -> None:
        logger.info("Client connected")
        turn_manager.recording_started_at = time.time()
        await audio_buffer.start_recording()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client) -> None:
        logger.info("Client disconnected")
        turn_manager.recording_ended_at = time.time()
        await save_transcript()
        await audio_buffer.stop_recording()
        await task.cancel()

    @audio_buffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = RECORDINGS_DIR / f"{timestamp}.wav"
        await save_audio_file(audio, filename, sample_rate, num_channels)

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(
        aggregator,
        strategy,
        message: UserTurnStoppedMessage,
    ) -> None:
        turn_manager.start_user_turn(timestamp=message.timestamp)
        turn_manager.end_user_turn(
            content=message.content,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency_seconds) -> None:
        turn_manager.start_assistant_turn(
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_seconds=latency_seconds,
        )

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(
        aggregator,
        message: AssistantTurnStoppedMessage,
    ) -> None:
        turn_manager.end_assistant_turn(
            content=message.content,
            is_interrupted=message.interrupted,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def save_transcript() -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        path = TRANSCRIPTS_DIR / f"{timestamp}.json"
        text = json.dumps(turn_manager.to_json(), indent=2)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(text)
        logger.info("Transcript saved to {}", path)

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point for the Pipecat runner."""
    transport: BaseTransport | None = None

    match runner_args:
        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection
            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            )
        case _:
            logger.error("Unsupported runner arguments type: {}", type(runner_args))
            return

    assert transport is not None
    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
