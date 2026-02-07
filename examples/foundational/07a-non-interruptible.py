#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
This example demonstrates how to dynamically toggle interruptions while still
transcribing user speech. Every 5 seconds, the bot toggles between:
- Interruptible mode: user speech will interrupt the bot
- Non-interruptible mode: user speech is transcribed but won't interrupt

This is useful when you want to capture what the user says during bot speech
without interrupting the bot's response, and then re-enable interruptions later.

The key mechanism is a custom frame (`EnableInterruptionsFrame`) that is queued
into the pipeline. A custom `DynamicVADUserTurnStartStrategy` subclasses
`VADUserTurnStartStrategy` and listens for this frame to update its
`enable_interruptions` setting at runtime.

In both modes:
- Voice Activity Detection (VAD) continues working
- Speech-to-text transcription continues
- User turns are aggregated into context

Watch the logs to see when interruptions are enabled/disabled, then try speaking
while the bot talks to observe the different behaviors.
"""

import asyncio
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMRunFrame, SystemFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies


@dataclass
class EnableInterruptionsFrame(SystemFrame):
    """A custom frame to dynamically enable or disable interruptions.

    Queue this frame into the pipeline to change the interruption behavior
    of DynamicVADUserTurnStartStrategy at runtime.
    """

    enable_interruptions: bool


class DynamicVADUserTurnStartStrategy(VADUserTurnStartStrategy):
    """A VAD-based user turn start strategy with dynamic interruption control.

    This strategy extends VADUserTurnStartStrategy to listen for
    EnableInterruptionsFrame, allowing the enable_interruptions setting
    to be changed at runtime via the pipeline.

    Example:
        # Create strategy with interruptions initially disabled
        strategy = DynamicVADUserTurnStartStrategy(enable_interruptions=False)

        # Later, toggle interruptions by queuing a frame
        await task.queue_frame(EnableInterruptionsFrame(enable_interruptions=True))
    """

    async def process_frame(self, frame: Frame):
        """Process frames, updating enable_interruptions when our custom frame arrives."""
        if isinstance(frame, EnableInterruptionsFrame):
            self._enable_interruptions = frame.enable_interruptions
            logger.info(f"Interruptions {'ENABLED' if frame.enable_interruptions else 'DISABLED'}")

        await super().process_frame(frame)


load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
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
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate toggling interruptible behavior. Give longer responses so the user can test speaking while you talk. Sometimes your speech can be interrupted, sometimes it cannot. The system will toggle every 5 seconds. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points.",
        },
    ]

    context = LLMContext(messages)

    # Create a dynamic VAD strategy that can be toggled at runtime via frames.
    # This strategy extends VADUserTurnStartStrategy but listens for
    # EnableInterruptionsFrame to change enable_interruptions dynamically.
    dynamic_strategy = DynamicVADUserTurnStartStrategy(enable_interruptions=False)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[dynamic_strategy],
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")

        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

        # Toggle interruptions every 5 seconds by queuing EnableInterruptionsFrame.
        # This is the idiomatic Pipecat way to control behavior via frames.
        interruptions_enabled = False
        for _ in range(10):  # Toggle 10 times (50 seconds total)
            await asyncio.sleep(5)
            interruptions_enabled = not interruptions_enabled

            # Queue a frame to toggle interruptions - the strategy will pick it up
            await task.queue_frame(
                EnableInterruptionsFrame(enable_interruptions=interruptions_enabled)
            )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
