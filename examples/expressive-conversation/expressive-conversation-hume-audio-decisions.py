#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.classifiers.base_classifier import ChoiceQuestion, YesNoQuestion
from pipecat.classifiers.hume.audio_decisions.classifier import HumeAudioClassifier
from pipecat.evals.transport import EvalTransportParams
from pipecat.expressive.expression import ExpressionDelivery, ExpressionSound
from pipecat.expressive.perception import format_perception
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy
from pipecat.expressive.processors.perception_reader import PerceptionQuestion, PerceptionReader
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.elevenlabs.dialogue.tts import ElevenLabsDialogueTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_turn_strategies import (
    FilterIncompleteUserTurnStrategies,
    MergeMode,
    UserTurnStrategiesParams,
)
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

    stt = DeepgramFluxSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    # Hume hears how each turn sounded: the user's tone of voice and whether
    # they sound unsure. The LLM reads it at the head of the user's message.
    # Hume hears at least 5 seconds of audio, so a short turn is heard with the
    # audio just before it, and with silence at the start of a call.
    reader = PerceptionReader(
        classifier=HumeAudioClassifier(api_key=os.environ["HUME_API_KEY"]),
        questions={
            "tone": PerceptionQuestion(
                question=ChoiceQuestion(
                    instructions="How does the speaker sound?",
                    options={
                        "neutral": None,
                        "happy": None,
                        "frustrated": None,
                        "angry": None,
                        "sad": None,
                        "confused": None,
                    },
                ),
                description="How the user sounded, heard from their voice.",
            ),
            "hesitation": PerceptionQuestion(
                question=YesNoQuestion(instructions="Does the speaker sound hesitant or unsure?"),
                description="The chance the user sounded unsure, from 0 to 1.",
            ),
        },
    )

    # Eleven v3 and v4 perform audio tags, which the Text-to-Dialogue service
    # serves.
    tts = ElevenLabsDialogueTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsDialogueTTSService.Settings(
            voice=os.environ["ELEVENLABS_VOICE_ID"],
            model="eleven_v4_turbo",
        ),
    )

    # With expressive conversation on, the LLM is taught how to read Hume's
    # readings and the expression tags the TTS performs. The policy's guidance
    # says what to do with a reading: matching the user's style changes how the
    # bot sounds, and the threshold lines change what it says.
    system_instruction = "You are a warm, expressive assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Keep your responses brief."

    expression_policy = ExpressionPolicy(
        guidance=[
            ExpressionGuidance.MATCH_STYLE,
            ExpressionGuidance.SLOW_FOR_DETAILS,
            "If the user sounds frustrated or angry at 0.7 or above, apologize once, then "
            "focus on helping.",
            "If hesitation is 0.7 or above, check that the user is following before you go on.",
        ],
        delivery=[
            ExpressionDelivery.SLOW,
            ExpressionDelivery.HAPPY,
            ExpressionDelivery.EXCITED,
            ExpressionDelivery.CURIOUS,
            ExpressionDelivery.THOUGHTFUL,
            ExpressionDelivery.WARM,
            ExpressionDelivery.SYMPATHETIC,
            ExpressionDelivery.REASSURING,
            ExpressionDelivery.NERVOUS,
            ExpressionDelivery.SAD,
            ExpressionDelivery.FRUSTRATED,
            ExpressionDelivery.SARCASTIC,
            ExpressionDelivery.DEADPAN,
            ExpressionDelivery.MISCHIEVOUS,
            ExpressionDelivery.WHISPERING,
        ],
        sounds=[
            ExpressionSound.LAUGH,
            ExpressionSound.CHUCKLE,
            ExpressionSound.GIGGLE,
            ExpressionSound.SIGH,
            ExpressionSound.EXHALE,
            ExpressionSound.LIGHT_BREATH,
            ExpressionSound.CLEAR_THROAT,
            ExpressionSound.GULP,
        ],
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(model="gpt-4.1", system_instruction=system_instruction),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # The reader sends Hume a clip each time the VAD hears a pause.
            vad_analyzer=SileroVADAnalyzer(),
            # Flux decides when the user stopped, and the LLM also judges
            # whether they finished, waiting for more when they trail off.
            user_turn_strategies=FilterIncompleteUserTurnStrategies(),
            user_turn_strategies_params=UserTurnStrategiesParams(stop_merge_mode=MergeMode.EXTEND),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            reader,  # How each user turn sounded
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
            # The LLM writes expression tags into its responses, and the TTS
            # performs them.
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
            logger.info(f"Hume: {format_perception(message.perception)}")

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
