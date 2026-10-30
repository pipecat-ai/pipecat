#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Run a voice bot with Sarvam speech models deployed on Amazon SageMaker.

Install the Sarvam SageMaker services with:

    uv add "pipecat-ai[sarvam,sagemaker]"

This example also uses Amazon Bedrock for its LLM. To run the complete example
outside a Pipecat development environment, install:

    uv add "pipecat-ai[aws,sarvam,sagemaker]"

Configure AWS credentials through the standard AWS credential chain. The
credentials must be able to invoke both SageMaker endpoints and the configured
Bedrock model.

Set these environment variables:

- ``AWS_REGION``: Region containing both SageMaker endpoints and the Bedrock model.
- ``SARVAM_SAGEMAKER_STT_ENDPOINT_NAME``: Endpoint serving the Saaras v3 API
  model ID ``saaras:v3``.
- ``SARVAM_SAGEMAKER_TTS_ENDPOINT_NAME``: Endpoint serving the Bulbul v3 model
  ID ``bulbul:v3``.

Audio assumptions:

- Saaras accepts WAV audio at 8 or 16 kHz. This example sends mono audio at 16 kHz.
- Bulbul returns mono ``linear16`` audio. This example requests 24 kHz output.
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
from pipecat.services.aws.llm import AWSBedrockLLMService, AWSBedrockLLMSettings
from pipecat.services.sarvam.sagemaker.stt import SarvamSageMakerSTTService
from pipecat.services.sarvam.sagemaker.tts import SarvamSageMakerTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=False)

STT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000

transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "livekit": lambda: LiveKitParams(
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
    """Run the Sarvam SageMaker voice bot."""
    logger.info("Starting bot")

    stt = SarvamSageMakerSTTService(
        endpoint_name=os.environ["SARVAM_SAGEMAKER_STT_ENDPOINT_NAME"],
        region=os.environ["AWS_REGION"],
        sample_rate=STT_SAMPLE_RATE,
        input_audio_codec="wav",
        settings=SarvamSageMakerSTTService.Settings(
            model="saaras:v3",
            language=Language.EN_IN,
            vad_signals=False,
        ),
    )

    tts = SarvamSageMakerTTSService(
        endpoint_name=os.environ["SARVAM_SAGEMAKER_TTS_ENDPOINT_NAME"],
        region=os.environ["AWS_REGION"],
        sample_rate=TTS_SAMPLE_RATE,
        output_audio_codec="linear16",
        settings=SarvamSageMakerTTSService.Settings(
            model="bulbul:v3",
            voice="shubh",
            language=Language.EN_IN,
            pace=1.0,
        ),
    )

    llm = AWSBedrockLLMService(
        aws_region=os.environ["AWS_REGION"],
        settings=AWSBedrockLLMSettings(
            model="us.amazon.nova-pro-v1:0",
            temperature=0.8,
            system_instruction=(
                "You are a helpful assistant in a voice conversation. "
                "Keep responses brief and suitable for spoken delivery."
            ),
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
            audio_in_sample_rate=STT_SAMPLE_RATE,
            audio_out_sample_rate=TTS_SAMPLE_RATE,
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
    """Create the selected transport and run the bot."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
