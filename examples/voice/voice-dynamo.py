#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Dynamo voice pipeline with incremental text prefill.

See the Dynamo section in examples/README.md for backend requirements and setup.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.dynamo.llm import DynamoRealtimeLLMService, DynamoStreamingPrefillProcessor
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv()

transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "websocket": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        serializer=ProtobufFrameSerializer(),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Dynamo voice bot")

    api_key = os.getenv("DYNAMO_API_KEY", "not-needed")
    stt = OpenAIRealtimeSTTService(
        api_key=api_key,
        base_url=os.getenv("DYNAMO_STT_BASE_URL", "ws://localhost:8000/v1/realtime"),
        settings=OpenAIRealtimeSTTService.Settings(model=os.environ["DYNAMO_STT_MODEL"]),
        turn_detection=False,
    )
    llm = DynamoRealtimeLLMService(
        api_key=api_key,
        base_url=os.getenv("DYNAMO_LLM_BASE_URL", "http://localhost:8000/v1"),
        settings=DynamoRealtimeLLMService.Settings(
            model=os.environ["DYNAMO_LLM_MODEL"],
            system_instruction=(
                "You are a helpful voice assistant. Reply briefly in plain spoken language, "
                "without markdown or other formatting that cannot be spoken."
            ),
            max_tokens=256,
        ),
    )
    tts = OpenAITTSService(
        api_key=api_key,
        base_url=os.getenv("DYNAMO_TTS_BASE_URL", "http://localhost:8000/v1"),
        settings=OpenAITTSService.Settings(
            model=os.environ["DYNAMO_TTS_MODEL"],
            voice=os.environ["DYNAMO_TTS_VOICE"],
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            # The user aggregator consumes interim transcripts, so prefill precedes it.
            DynamoStreamingPrefillProcessor(llm=llm, context=context),
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=24000,
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
        context.add_message({"role": "user", "content": "Please introduce yourself."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Run the pipeline with the development runner's selected transport."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
