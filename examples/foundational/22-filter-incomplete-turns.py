#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndTaskFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

load_dotenv(override=True)


# System prompts for idle follow-ups
IDLE_PROMPT_FIRST = """The user has been quiet for a moment. Based on the conversation context, generate a brief, natural follow-up to re-engage them. Your response should:
- Be contextually relevant to what was just discussed
- Sound natural and conversational (not robotic)
- Be concise (1-2 sentences max)
- Gently prompt them to continue without being pushy

Examples:
- If you asked a question: "Take your time! I'm curious to hear your thoughts."
- If discussing a topic: "What do you think about that?"
- If they seemed engaged: "Are you still there? I'd love to hear more."

Generate ONLY the follow-up message, nothing else."""

IDLE_PROMPT_SECOND = """The user has been quiet for a while now. Generate a brief, friendly check-in message. Your response should:
- Acknowledge they might be busy or thinking
- Offer to continue or pause the conversation
- Be warm and understanding
- Be very brief (1 sentence)

Examples:
- "No rush! Let me know if you'd like to continue."
- "Take your time - I'm here when you're ready."
- "Should we pick this up later?"

Generate ONLY the check-in message, nothing else."""


class IdleHandler:
    """Helper class to manage user idle retry logic with contextually-aware LLM responses.

    This handler pushes messages through the pipeline to generate contextually appropriate
    follow-up messages when a user becomes idle. Using the pipeline ensures that
    interruptions are handled automatically if the user starts speaking.
    """

    def __init__(self):
        """Initialize the idle handler."""
        self._retry_count = 0

    def reset(self):
        """Reset the retry count when user becomes active."""
        self._retry_count = 0

    async def handle_idle(self, aggregator):
        """Handle user idle event with escalating prompts.

        Pushes a system message and triggers LLM through the pipeline, allowing
        normal interruption handling if the user starts speaking.

        Args:
            aggregator: The user aggregator that triggered the idle event.
        """
        self._retry_count += 1
        logger.info(f"Handling idle event (attempt {self._retry_count})")

        if self._retry_count <= 2:
            # Select the appropriate prompt based on retry count
            prompt = IDLE_PROMPT_FIRST if self._retry_count == 1 else IDLE_PROMPT_SECOND

            # Push through pipeline - this allows interruption handling
            await aggregator.push_frame(
                LLMMessagesAppendFrame(messages=[{"role": "system", "content": prompt}])
            )
            await aggregator.push_frame(LLMRunFrame())
        else:
            # Third attempt: End the conversation gracefully
            logger.info("User idle timeout reached, ending conversation")
            await aggregator.push_frame(
                TTSSpeakFrame("It seems like you're busy right now. Have a nice day!")
            )
            await aggregator.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way.""",
        }
    ]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            ),
            user_idle_timeout=6.0,
            filter_incomplete_user_turns=True,
        ),
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Initialize idle handler for contextual responses through the pipeline
    idle_handler = IdleHandler()

    @user_aggregator.event_handler("on_user_turn_idle")
    async def on_user_turn_idle(aggregator):
        await idle_handler.handle_idle(aggregator)

    @user_aggregator.event_handler("on_user_turn_started")
    async def handle_user_turn_started(aggregator, strategy):
        idle_handler.reset()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append(
            {
                "role": "system",
                "content": "Please introduce yourself to the user, asking them a question that will require a complete response. To start, say 'Let me start with a fun one. If you could travel anywhere in the world right now, where would you go and why?'",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
