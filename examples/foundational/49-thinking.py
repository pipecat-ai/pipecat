#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os
import random
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, ThoughtTranscriptionMessage, TranscriptionMessage
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# LLM provider constants
LLM_ANTHROPIC = "anthropic"
LLM_GOOGLE = "google"
LLM_DEFAULT = LLM_GOOGLE

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(
    transport: BaseTransport, runner_args: RunnerArguments, llm_provider: str = LLM_DEFAULT
):
    logger.info(f"Starting bot with {llm_provider.capitalize()} LLM")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    if llm_provider == LLM_ANTHROPIC:
        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            params=AnthropicLLMService.InputParams(
                thinking=AnthropicLLMService.ThinkingConfig(type="enabled", budget_tokens=2048)
            ),
        )
    elif llm_provider == LLM_GOOGLE:
        llm = GoogleLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            # model="gemini-3-pro-preview", # A more powerful reasoning model, but slower
            params=GoogleLLMService.InputParams(
                thinking=GoogleLLMService.ThinkingConfig(
                    # thinking_level="low", # Use this field instead of thinking_budget for Gemini 3 Pro. Defaults to "high".
                    thinking_budget=-1,  # Dynamic thinking
                    include_thoughts=True,
                )
            ),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    transcript = TranscriptProcessor()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            transcript.user(),  # User transcripts
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            transcript.assistant(),  # Assistant transcripts (including thoughts)
            context_aggregator.assistant(),  # Assistant spoken responses
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
        # Choose a random prompt to demonstrate thinking capabilities.
        # These prompts were chosen from Google and Anthropic docs.
        thinking_prompt_1 = "Analogize photosynthesis and growing up."
        thinking_prompt_2 = "Compare and contrast electric cars and hybrid cars."
        thinking_prompt_3 = "Are there an infinite number of prime numbers such that n mod 4 == 3?"
        selected_prompt = random.choice([thinking_prompt_1, thinking_prompt_2, thinking_prompt_3])

        # Kick off the conversation.
        messages.append({"role": "user", "content": selected_prompt})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    # Register event handler for transcript updates
    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        for msg in frame.messages:
            if isinstance(msg, (ThoughtTranscriptionMessage, TranscriptionMessage)):
                timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                role = "THOUGHT" if isinstance(msg, ThoughtTranscriptionMessage) else msg.role
                logger.info(f"Transcript: {timestamp}{role}: {msg.content}")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    # Get llm_provider from module attribute set in __main__
    llm_provider = getattr(sys.modules[__name__], "llm_provider", LLM_DEFAULT)
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args, llm_provider)


if __name__ == "__main__":
    # Parse custom arguments before calling runner main()
    parser = argparse.ArgumentParser(description="Thinking LLM Bot")
    parser.add_argument(
        "--llm",
        type=str,
        choices=[LLM_ANTHROPIC, LLM_GOOGLE],
        default=LLM_DEFAULT,
        help=f"LLM provider to use (default: {LLM_DEFAULT})",
    )
    # Parse only known args to allow runner's main() to handle its own args
    args, remaining = parser.parse_known_args()

    # Store the llm_provider in sys.modules for bot() function to access
    sys.modules[__name__].llm_provider = args.llm

    # Restore sys.argv with remaining args for runner's main()
    sys.argv[1:] = remaining

    from pipecat.runner.run import main

    main()
