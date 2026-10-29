#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice conversation using Fish realtime ASR and Fish TTS.

Requires FISH_API_KEY, FISH_VOICE_ID, and OPENAI_API_KEY. Run with the dev
runner, for example: ``python voice-fish-realtime.py -t webrtc``.
"""

import os

from dotenv import load_dotenv

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.fish.stt import FishAudioSTTService
from pipecat.services.fish.tts import FishAudioTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run a conversation with turn detection provided by Fish."""
    stt = FishAudioSTTService(api_key=os.environ["FISH_API_KEY"])
    tts = FishAudioTTSService(
        api_key=os.environ["FISH_API_KEY"],
        settings=FishAudioTTSService.Settings(voice=os.environ["FISH_VOICE_ID"]),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction="You are a helpful voice assistant. Keep your replies brief.",
        ),
    )
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
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
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        context.add_message({"role": "developer", "content": "Greet the user."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Create the selected transport and run the bot."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
