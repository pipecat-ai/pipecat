#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A bot in a conversation between several people, which speaks only when it's spoken to.

Pipecat Sidechat (experimental) decides whether each user turn was for the bot.
``PipecatSidechat`` lets the LLM answer a turn meant for the bot, answers a
question to the room if nobody else does, and keeps the bot quiet otherwise.
The bot still hears every turn.

Try it with two people in one room: ask "Atlas" something, ask a follow-up
without its name, then talk among yourselves.
"""

import os

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
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.sidechat import SIDECHAT_SPEAKER_FORMAT, PipecatSidechat, SidechatVerdict
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

BOT_NAME = "Atlas"

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "livekit": lambda: LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    # Diarization tags each transcript with its speaker, which is how Sidechat
    # and the LLM know who said what. The keyterm keeps the bot's name from
    # being misheard.
    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTService.Settings(
            diarize=True, speaker_format=SIDECHAT_SPEAKER_FORMAT, keyterm=[BOT_NAME]
        ),
    )

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(voice="86e30c1d-714b-4074-a1f2-1cb6b552fb49"),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                f"You are {BOT_NAME}, an AI assistant on a call with several people. Each "
                "part of a message starts with its speaker's tag in square brackets; never "
                "read the tags aloud. Reply to the latest message, briefly. Your responses "
                "will be spoken aloud, so avoid emojis, bullet points, or other formatting "
                "that can't be spoken."
            ),
        ),
    )

    # The model downloads from Hugging Face the first time.
    sidechat = PipecatSidechat(
        agent_name=BOT_NAME,
        # What the bot is and does, in a line, when it isn't an assistant on
        # a call. The model reads it with every turn.
        # role="voice assistant in the meeting room's screen; it shows the agenda",
        # What people ask this bot for without its name, as they say it. A turn
        # that contains one is answered whatever the model reads, so leave out
        # words people also use with each other.
        # commands=["show the agenda", "next item"],
        # The small model reads a turn in a few milliseconds. The base model
        # decides better, most of all on requests that don't name the bot,
        # and takes a few hundred milliseconds on one CPU.
        # variant="base",
        # The full weights. The int8 default is smaller and faster, and decides
        # the same on nearly every line.
        # precision="fp32",
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
            sidechat,  # Lets the LLM answer only the turns for the bot
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

    @sidechat.event_handler("on_sidechat_decision")
    async def on_sidechat_decision(sidechat, verdict: SidechatVerdict):
        logger.info(f"{verdict.decision} ({verdict.reason}): {verdict.line.text}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message({"role": "developer", "content": "Introduce yourself in a sentence."})
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
