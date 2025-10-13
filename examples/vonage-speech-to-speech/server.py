# SPDX-License-Identifier: BSD-2-Clause
"""Speech↔Speech via OpenAI Realtime (no separate STT/TTS)."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, StartInterruptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.vonage import VonageFrameSerializer

# Realtime S2S (audio-in/audio-out) service
from pipecat.services.openai_realtime_beta.context import OpenAIRealtimeLLMContext
from pipecat.services.openai_realtime_beta.openai import OpenAIRealtimeBetaLLMService
from pipecat.transports.network.websocket_server import WebsocketServerParams
from pipecat.transports.vonage.audio_connector import VonageAudioConnectorTransport

WS_HOST = "0.0.0.0"
WS_PORT = 8005
SESSION_TIMEOUT_SECONDS = 60 * 3
AUDIO_OUT_SAMPLE_RATE = 16_000  # telephony-friendly

SYSTEM_INSTRUCTION = (
    "You are a concise, friendly voice assistant. "
    "You will receive spoken input and respond with speech. "
    "Always respond in ENGLISH only, even if the user speaks another language. "
    "Keep replies to one or two sentences and avoid special characters."
)

load_dotenv()


# Cancels the Realtime model when user starts speaking (barge-in).
class RealtimeBargeInCanceler(FrameProcessor):
    def __init__(self, realtime_service):
        super().__init__()
        self._realtime = realtime_service

    # Direction-aware forwarding to avoid feedback loops.
    async def queue_frame(self, frame: Frame, direction):
        # Only cancel on *downstream* interruption (from mic/user)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, StartInterruptionFrame):
            cancelled = False
            for method_name in (
                "cancel_current_response",
                "cancel_response",
                "stop_current_response",
            ):
                try:
                    method = getattr(self._realtime, method_name, None)
                    if method:
                        await method()
                        cancelled = True
                        break
                except Exception as e:
                    logger.warning(f"Realtime cancel via {method_name} failed: {e}")
            if not cancelled:
                logger.warning(
                    "Realtime cancel method not found; barge-in will rely on VAD + clearAudio only."
                )

        # Forward respecting direction to prevent recursion
        if direction == FrameDirection.DOWNSTREAM:
            if self._next:
                await self._next.queue_frame(frame, direction)
        else:  # UPSTREAM
            if self._prev:
                await self._prev.queue_frame(frame, direction)


async def run_bot_websocket_server() -> None:
    serializer = VonageFrameSerializer()

    # VAD tuned for barge-in (times in seconds)
    vad = SileroVADAnalyzer(
        sample_rate=AUDIO_OUT_SAMPLE_RATE,
        params=VADParams(
            confidence=0.7,
            start_secs=0.12,  # ~120 ms to declare speaking
            stop_secs=0.25,  # ~250 ms silence to stop
            min_volume=0.6,
        ),
    )

    ws_transport = VonageAudioConnectorTransport(
        host=WS_HOST,
        port=WS_PORT,
        params=WebsocketServerParams(
            serializer=serializer,
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_analyzer=vad,
            session_timeout=SESSION_TIMEOUT_SECONDS,
        ),
    )

    realtime = OpenAIRealtimeBetaLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-realtime-preview-2025-06-03",
        send_transcription_frames=False,
        # Optional knobs if supported:
        # transcription_language="en",
        # enable_server_vad=True,
        # max_output_chunk_ms=200,
    )

    canceler = RealtimeBargeInCanceler(realtime)

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    context = OpenAIRealtimeLLMContext(messages)
    context_agg = realtime.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            ws_transport.input(),  # audio from Vonage over WS
            canceler,  # cancel model on StartInterruptionFrame (direction-aware)
            context_agg.user(),  # seed system context once
            realtime,  # audio-in/audio-out model
            ws_transport.output(),  # audio back to Vonage
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=AUDIO_OUT_SAMPLE_RATE,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(_t, _c):
        logger.info("Client connected")
        # Send the system context after everything is linked and running
        await task.queue_frames([context_agg.user().get_context_frame()])

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_t, _c):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(run_bot_websocket_server())
