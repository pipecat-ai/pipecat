#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voicemail detection that can resume into a live conversation.

The built-in ``VoicemailDetector`` is designed for "detect voicemail, leave a
message, hang up": once it classifies the call as voicemail it closes its
``ConversationGate`` permanently, so the main LLM never hears anything else.

This example shows the small amount of custom state management needed to keep
the call going if a human picks up *after* the message starts. When voicemail is
detected we leave the message, then reopen the detector's conversation path so
that the next thing the called party says flows back to the main LLM and the bot
continues a normal conversation.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.extensions.voicemail.voicemail_detector import VoicemailDetector
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
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

# The message the bot leaves when it reaches a voicemail system. It invites the
# called party to jump in, which is what the resume path below handles.
VOICEMAIL_MESSAGE = (
    "Hi, this is Jamie calling about your appointment tomorrow. "
    "Please call me back at 555-0123. If you're there, feel free to jump in."
)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are Jamie, a friendly assistant making an outbound phone call about "
                "the person's appointment tomorrow. Your responses will be spoken aloud, so "
                "avoid emojis, bullet points, or other formatting that can't be spoken. If the "
                "person you called speaks to you, talk with them naturally and briefly about "
                "their appointment."
            ),
        ),
    )
    classifier_llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"])

    voicemail = VoicemailDetector(llm=classifier_llm)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            voicemail.detector(),  # Voicemail detection — between STT and User context aggregator
            user_aggregator,
            llm,
            tts,
            voicemail.gate(),  # TTS gating — Immediately after the TTS service
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

    @voicemail.event_handler("on_conversation_detected")
    async def on_conversation_detected(processor):
        logger.info("Conversation detected — a human answered, talking normally.")

    @voicemail.event_handler("on_voicemail_detected")
    async def on_voicemail_detected(processor):
        logger.info("Voicemail detected! Leaving a message...")

        # Leave the message. This is pushed straight to TTS, bypassing the main
        # LLM, just like the standard voicemail example.
        await processor.push_frame(TTSSpeakFrame(VOICEMAIL_MESSAGE))

        # Resume support (the custom state management this example is about):
        #
        # By default the detector closes its ConversationGate permanently once
        # voicemail is detected, which blocks the main LLM from ever hearing the
        # called party again. We reopen that gate so that if a human picks up
        # and starts talking after the message, their speech flows back into the
        # main pipeline and the bot continues a normal conversation.
        #
        # The classifier branch stays closed (the call is already classified),
        # so reopening only restores the live-conversation path.
        voicemail._conversation_gate._gate_opened = True
        logger.info("Conversation path reopened — the called party can keep talking.")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
