#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
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
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.services.speechmatics.tts import SpeechmaticsTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run example using Speechmatics STT and TTS.

    This example demonstrates a complete Speechmatics integration with both Speech-to-Text
    and Text-to-Speech services:

    STT Features:
    - Diarization to identify and distinguish between different speakers
    - Words spoken by each speaker are wrapped with XML tags for LLM processing
    - System context instructions help the LLM understand multi-party conversations
    - ENHANCED operating point by default for optimal accuracy

    TTS Features:
    - Low latency streaming audio synthesis
    - Multiple voice options available including `sarah`, `theo`, `megan` and `jack`

    For more information:
    - STT: https://docs.speechmatics.com/rt-api-ref
    - TTS: https://docs.speechmatics.com/text-to-speech/quickstart
    """
    logger.info(f"Starting bot")

    async with aiohttp.ClientSession() as session:
        stt = SpeechmaticsSTTService(
            api_key=os.getenv("SPEECHMATICS_API_KEY"),
            params=SpeechmaticsSTTService.InputParams(
                language=Language.EN,
                speaker_active_format="<{speaker_id}>{text}</{speaker_id}>",
            ),
        )

        tts = SpeechmaticsTTSService(
            api_key=os.getenv("SPEECHMATICS_API_KEY"),
            voice_id="sarah",
            aiohttp_session=session,
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            params=BaseOpenAILLMService.InputParams(temperature=0.75),
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful British assistant called Sarah. "
                    "Your goal is to demonstrate your capabilities in a succinct way. "
                    "Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. "
                    "Always include punctuation in your responses. "
                    "Give very short replies - do not give longer replies unless strictly necessary. "
                    "Respond to what the user said in a concise, funny, creative and helpful way. "
                    "Use `<Sn/>` tags to identify different speakers - do not use tags in your replies."
                ),
            },
        ]

        context = LLMContext(messages)
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
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
            messages.append({"role": "system", "content": "Say a short hello to the user."})
            await task.queue_frames([LLMRunFrame()])

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
