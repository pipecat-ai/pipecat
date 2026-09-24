#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Empty User Turn Recovery

A user turn can start on voice activity alone and end with no transcript: the
user coughed, spoke over background noise, or the STT simply missed it. When
that turn interrupted the bot, the bot has stopped talking and the LLM has
nothing new to answer, so the conversation stalls.

`EmptyUserTurnConfig` makes the user aggregator run the LLM once for such a
turn, with a developer message saying that the user spoke but wasn't
understood. The bot then asks the user to repeat, or repeats what they may
have missed. An empty turn while the bot was idle is most likely noise, and is
left unanswered by default; `user_idle_timeout` still applies after it, so the
bot checks in if the user then stays quiet.

To try it without a noisy room, say "pineapple": this bot drops any transcript
containing that word, as if the STT had failed to recognize it. Ask for a long
story, then say "pineapple" while the bot is talking.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.empty_user_turn import EmptyUserTurnConfig
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# Transcripts containing this word are dropped, as if never recognized.
UNRECOGNIZED_WORD = "pineapple"


class UnrecognizedSpeechSimulator(FrameProcessor):
    """Drops transcripts containing `UNRECOGNIZED_WORD`."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if UNRECOGNIZED_WORD in frame.text.lower():
                logger.info(f"Dropping transcript to simulate unrecognized speech: {frame.text}")
                return

        await self.push_frame(frame, direction)


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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
    )

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="86e30c1d-714b-4074-a1f2-1cb6b552fb49",
        ),
    )

    context = LLMContext()
    # An empty turn is closed by the `user_turn_stop_timeout` watchdog (5s by
    # default), so the recovery comes that long after the user stops talking.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_idle_timeout=8.0,
            empty_user_turn=EmptyUserTurnConfig(
                # Optional: customize the recovery
                # interrupted_prompt="Custom prompt...",
                # idle_prompt="The user may have said something that wasn't understood...",
                # max_consecutive_recoveries=2,
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            UnrecognizedSpeechSimulator(),  # Drop unrecognized speech, only for demo
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

    @user_aggregator.event_handler("on_user_turn_idle")
    async def on_user_turn_idle(aggregator):
        logger.info("User turn idle")
        # The user aggregator consumes `LLMMessagesAppendFrame`, so push from it
        # to reach the LLM service downstream.
        await aggregator.push_frame(
            LLMMessagesAppendFrame(
                [
                    {
                        "role": "developer",
                        "content": (
                            "The user has been quiet. Politely and briefly ask if they're "
                            "still there."
                        ),
                    }
                ],
                run_llm=True,
            )
        )

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}user: {message.content}"
        logger.info(f"Transcript: {line}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}assistant: {message.content}"
        logger.info(f"Transcript: {line}")

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
