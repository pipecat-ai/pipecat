#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    StartInterruptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver, FrameEndpoint
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


class CustomObserver(BaseObserver):
    """Observer to log interruptions and bot speaking events to the console.

    Logs all frame instances of:
    - StartInterruptionFrame
    - BotStartedSpeakingFrame
    - BotStoppedSpeakingFrame

    This allows you to see the frame flow from processor to processor through the pipeline for these frames.
    Log format: [EVENT TYPE]: [source processor] → [destination processor] at [timestamp]s
    """

    async def on_push_frame(self, data: FramePushed):
        src = data.source
        dst = data.destination
        frame = data.frame
        direction = data.direction
        timestamp = data.timestamp

        # Convert timestamp to seconds for readability
        time_sec = timestamp / 1_000_000_000

        # Create direction arrow
        arrow = "→" if direction == FrameDirection.DOWNSTREAM else "←"

        if isinstance(frame, StartInterruptionFrame) and isinstance(src, BaseOutputTransport):
            logger.info(f"⚡ INTERRUPTION START: {src} {arrow} {dst} at {time_sec:.2f}s")
        elif isinstance(frame, BotStartedSpeakingFrame):
            logger.info(f"🤖 BOT START SPEAKING: {src} {arrow} {dst} at {time_sec:.2f}s")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.info(f"🤖 BOT STOP SPEAKING: {src} {arrow} {dst} at {time_sec:.2f}s")


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[
            CustomObserver(),
            LLMLogObserver(),
            DebugLogObserver(
                frame_types={
                    TTSTextFrame: (BaseOutputTransport, FrameEndpoint.DESTINATION),
                    UserStartedSpeakingFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
                    EndFrame: None,
                }
            ),
        ],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
