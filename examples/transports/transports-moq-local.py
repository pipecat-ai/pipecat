#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MOQ (Media over QUIC) transport example — fully local, no API keys.

Same pipeline as ``transports-moq.py`` but every AI service runs locally,
so no Deepgram / Cartesia / OpenAI keys are needed:

    - STT: MLX Whisper (Apple Silicon)
    - LLM: Ollama (expects a model already pulled, e.g. ``ollama pull llama3.2:1b``)
    - TTS: Kokoro (local ONNX, auto-downloads model files on first use)

Usage:
    uv run python examples/transports/transports-moq-local.py \\
        -t moq --moq-serve --moq-tls-generate localhost
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
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
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.moq.transport import MOQParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# Local model selection — overridable via env so you don't have to edit code.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")

# Transport-specific parameters using lambdas for deferred creation
transport_params = {
    "moq": lambda: MOQParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the bot with the given transport."""
    logger.info("Starting bot (local services)")

    stt = WhisperSTTServiceMLX(model=MLXModel.LARGE_V3_TURBO_Q4.value)

    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=KOKORO_VOICE),
    )

    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model=OLLAMA_MODEL,
            system_instruction=(
                "You are a helpful assistant in a real-time voice call. "
                "Your goal is to demonstrate your capabilities in a succinct way. "
                "Your output will be spoken aloud, so avoid special characters that can't easily "
                "be spoken, such as emojis or bullet points. Respond to what the user said in a "
                "creative and helpful way."
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
            transport.input(),  # Transport user input
            stt,
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport):
        logger.info("Client subscribed — starting conversation")
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_disconnected")
    async def on_disconnected(transport):
        logger.info("Disconnected from MOQ relay")
        await worker.cancel()

    @transport.event_handler("on_error")
    async def on_error(transport, message, exception):
        logger.error(f"MOQ error: {message}")

    # MOQInputTransport.start() auto-connects to the relay when the
    # pipeline starts, so we don't dial transport.connect() here.
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        await transport.disconnect()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
