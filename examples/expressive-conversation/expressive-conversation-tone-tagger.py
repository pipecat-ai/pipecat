#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.expressive.perception import format_perception
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.processors.tone.reader import PipecatToneReader
from pipecat.processors.tone.tagger import PipecatToneTagger
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.dialogue.tts import ElevenLabsDialogueTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
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
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    # Pipecat Tone reads how each turn was meant from its words: the user's
    # emotion and whether they're sarcastic. The LLM reads it at the head of
    # the user's message. The model downloads from Hugging Face at start.
    tone_reader = PipecatToneReader()

    # The LLM writes plain words, and Pipecat Tone's tagger voices them: for
    # each sentence it chooses a delivery, slow spans for details, stressed
    # words and sounds, from the call so far and how the user sounded. Its
    # model downloads from Hugging Face at start.
    tone_tagger = PipecatToneTagger()

    # Eleven v3 and v4 perform audio tags, which the Text-to-Dialogue service
    # serves.
    tts = ElevenLabsDialogueTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsDialogueTTSService.Settings(
            voice=os.environ["ELEVENLABS_VOICE_ID"],
            model="eleven_v4_turbo",
        ),
    )

    # With a tagger in the pipeline, the LLM is taught how to read Tone's
    # readings but no tags, and the context keeps its replies plain. The
    # presets steer the tagger; the policy's own lines say what the LLM does
    # with a reading.
    system_instruction = "You are a warm assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Keep your responses brief."

    expression_policy = ExpressionPolicy(
        guidance=[
            ExpressionGuidance.MATCH_STYLE,
            ExpressionGuidance.SLOW_FOR_DETAILS,
            ExpressionGuidance.BREATHE,
            "If the user sounds angry at 0.7 or above, apologize once, then focus on helping.",
            "If sarcasm is 0.7 or above, the user means the opposite of their words: "
            "answer what they mean.",
        ],
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-5.6-luna", system_instruction=system_instruction
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
            tone_reader,  # How each user turn was meant
            user_aggregator,  # User responses
            llm,  # LLM
            tone_tagger,  # How each sentence is voiced
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
            # The tagger writes expression tags into the LLM's responses, and
            # the TTS performs them.
            enable_expressive_conversation=True,
            expression_policy=expression_policy,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        if message.perception:
            logger.info(f"Tone: {format_perception(message.perception)}")

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
