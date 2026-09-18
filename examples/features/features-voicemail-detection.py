#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Outbound-call bot that tells a live person from voicemail with TypeSafe.

The bot places (or, here, receives) a call and needs to know who picked up
before it says anything. ``VoicemailDetector`` holds the bot's speech back
while a classifier judges the first thing the other side says. The classifier
is a ``TypeSafeVoicemailClassifier``: one ``Choice`` per caller turn, answered in
about a fifth of a second, with no text generated. A person gets the normal
LLM conversation; a recording gets a written message once it goes quiet.

Requirements:
- TYPESAFE_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- OPENAI_API_KEY (the conversation LLM; not used for detection)

Run the example:
uv run examples/features/features-voicemail-detection.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.extensions.voicemail.typesafe_classifier import TypeSafeVoicemailClassifier
from pipecat.extensions.voicemail.voicemail_detector import VoicemailDetector
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
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
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
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

VOICEMAIL_MESSAGE = (
    "Hello, this is Jamie calling about your appointment. "
    "Please call me back at 555-0123 when you get this."
)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="86e30c1d-714b-4074-a1f2-1cb6b552fb49",
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
    )

    # The classifier LLM is a TypeSafe judgment: a live person, or a recording? It
    # answers nothing below confidence 0.5, so the detector waits and a lone
    # "sorry, I can't come to the phone right now" is judged again together
    # with whatever follows it.
    voicemail = VoicemailDetector(llm=TypeSafeVoicemailClassifier())

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            voicemail.detector(),  # Voicemail detection: between STT and the user context aggregator
            user_aggregator,
            llm,
            tts,
            voicemail.gate(),  # TTS gating: immediately after the TTS service
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    @voicemail.event_handler("on_conversation_detected")
    async def on_conversation_detected(processor):
        logger.info("Conversation detected!")

    @voicemail.event_handler("on_voicemail_detected")
    async def on_voicemail_detected(processor):
        logger.info("Voicemail detected! Leaving a message...")

        # Push frames using standard Pipecat pattern
        await processor.push_frame(TTSSpeakFrame(VOICEMAIL_MESSAGE))

        # NOTE: A common pattern is to end pipeline after the voicemail is left.
        # Uncomment the following line to end the pipeline after leaving the voicemail.
        # await processor.push_frame(EndWorkerFrame())

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
