#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Interruptible voice bot using Krisp streaming STT and TTS.

This example demonstrates a conversational bot with:
- Krisp real-time STT, whose end-of-turn prediction ends user turns
- Silero VAD starting user turns (and interrupting the bot)
- Krisp streaming TTS

No ``user_turn_strategies`` are passed to the user aggregator, so it uses the
strategies ``KrispSTTService`` recommends: pipecat's default turn start and
Krisp's end of turn.

Run locally and open http://localhost:7860/client in a browser:

    python voice-krisp-stt-tts.py

Required environment variables:
- KRISP_API_KEY: Krisp master API key (session tokens are created from it)
- OPENAI_API_KEY: OpenAI API key for LLM

Optional environment variables:
- KRISP_TTS_LANGUAGE: language the bot speaks, e.g. de-DE (default: en-US). It
  is also the STT language hint and the language the LLM is asked to reply in.
- KRISP_TTS_VOICE: Krisp TTS voice name. Validated against the voice catalog for
  KRISP_TTS_LANGUAGE; when unset, a catalog voice for that language is used.
- KRISP_TTS_GENDER: female, male or neutral; narrows the automatic voice choice.
- KRISP_REST_BASE_URL, KRISP_STT_URL, KRISP_TTS_URL: endpoint overrides, e.g.
  for staging: https://stage-api.developers.krisp.ai,
  wss://stage.streaming.krisp.ai/stt and wss://stage.streaming.krisp.ai/tts
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.krisp.api import KRISP_REST_BASE_URL, select_tts_voice
from pipecat.services.krisp.stt import KrispSTTService
from pipecat.services.krisp.tts import KrispTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    rest = (
        {"rest_base_url": os.environ["KRISP_REST_BASE_URL"]}
        if "KRISP_REST_BASE_URL" in os.environ
        else {}
    )
    stt_url = {"url": os.environ["KRISP_STT_URL"]} if "KRISP_STT_URL" in os.environ else {}
    tts_url = {"url": os.environ["KRISP_TTS_URL"]} if "KRISP_TTS_URL" in os.environ else {}

    # Pick the voice from the catalog before creating the TTS service, so the
    # voice and language passed to its handshake are a valid combination.
    language = os.getenv("KRISP_TTS_LANGUAGE", "en-US")
    voice = await select_tts_voice(
        os.environ["KRISP_API_KEY"],
        language,
        voice=os.getenv("KRISP_TTS_VOICE"),
        gender=os.getenv("KRISP_TTS_GENDER"),
        base_url=rest.get("rest_base_url", KRISP_REST_BASE_URL),
    )
    logger.info(f"Krisp TTS voice: {voice} ({language})")

    stt = KrispSTTService(
        api_key=os.environ["KRISP_API_KEY"],
        settings=KrispSTTService.Settings(language=language),
        **rest,
        **stt_url,
    )

    tts = KrispTTSService(
        api_key=os.environ["KRISP_API_KEY"],
        settings=KrispTTSService.Settings(voice=voice, language=language),
        **rest,
        **tts_url,
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=f"You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way. Always reply in the language {language}.",
        ),
    )

    context = LLMContext()
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

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[MetricsLogObserver(include_metrics={TurnMetricsData})],
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
