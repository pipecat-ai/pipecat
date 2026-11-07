#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Several TTSSpeakFrames pushed at once, each synthesized in its own context.

A TTS service that doesn't hold frames per turn opens a context for each
`TTSSpeakFrame` as it arrives, so the three sentences below synthesize
concurrently on one socket and still play in order. The bot speaks a burst on
connect and another when the user asks for the list (the `read_list` tool).

Pick the TTS with the runner body: `{"tts": "cartesia"}` (the default) or
`{"tts": "elevenlabs"}`.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import FunctionCallResultProperties, TTSSpeakFrame
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
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

GREETING = [
    "First, the striped zebra crossed the wide muddy river at dawn.",
    "Second, the old lighthouse keeper polished every brass lamp in the tower.",
    "Third, a quiet violinist tuned her strings before the evening concert.",
]

LIST = [
    "One, a basket of ripe orange tangerines.",
    "Two, a heavy wool blanket for the cabin.",
    "Three, a box of matches and a lantern.",
]


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


def create_tts(name: str) -> TTSService:
    if name == "elevenlabs":
        return ElevenLabsTTSService(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            settings=ElevenLabsTTSService.Settings(
                voice=os.getenv("ELEVENLABS_VOICE_ID", ""),
            ),
        )
    return CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="86e30c1d-714b-4074-a1f2-1cb6b552fb49",
        ),
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    tts_name = (runner_args.body or {}).get("tts", "cartesia")
    logger.info(f"Starting bot with {tts_name} TTS")

    stt = OpenAIRealtimeSTTService(api_key=os.environ["OPENAI_API_KEY"])

    tts = create_tts(tts_name)

    async def speak_all(sentences: list[str]):
        # Queued together, so the TTS has all of them before the first one plays.
        for sentence in sentences:
            await tts.queue_frame(TTSSpeakFrame(sentence))

    async def read_list(params: FunctionCallParams):
        """Read the packing list aloud to the user."""
        await speak_all(LIST)
        await params.result_callback(
            {"status": "read aloud"},
            properties=FunctionCallResultProperties(run_llm=False),
        )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. When the user asks for the list, call read_list and say nothing else.",
        ),
    )

    context = LLMContext(tools=[read_list])
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
        await speak_all(GREETING)

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
