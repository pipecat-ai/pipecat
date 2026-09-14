#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Korean voice conversation with Airy TTS and OpenAI STT/LLM.

Set AIRY_API_KEY and OPENAI_API_KEY in .env, then run::

    uv run --extra airy --extra runner --extra webrtc examples/voice/voice-airy.py

Open http://localhost:7860/client to talk to the bot.
"""

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.airy.tts import AiryHttpTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run a Korean conversation through Airy's streaming TTS service."""
    stt = OpenAISTTService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAISTTService.Settings(language=Language.KO),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a helpful voice assistant. Reply in Korean, in one or two short "
                "sentences. Your replies will be spoken aloud, so avoid Markdown and emojis."
            ),
        ),
    )
    user, assistant = LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    async with aiohttp.ClientSession() as session:
        tts = AiryHttpTTSService(
            api_key=os.environ["AIRY_API_KEY"],
            aiohttp_session=session,
            settings=AiryHttpTTSService.Settings(language=Language.KO),
        )
        worker = PipelineWorker(
            Pipeline([transport.input(), stt, user, llm, tts, transport.output(), assistant]),
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )
        runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
        await runner.add_workers(worker)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            user.context.add_message({"role": "user", "content": "안녕하세요."})
            await worker.queue_frame(LLMRunFrame())

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await runner.cancel()

        await runner.run()


async def bot(runner_args: RunnerArguments):
    """Create the selected transport and run the bot."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
