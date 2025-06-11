#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import sys
from typing import Any, Dict

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecatcloud.agent import DailySessionArguments

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
)
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

# Check if we're in local development mode
LOCAL_RUN = os.getenv("LOCAL_RUN")

logger.add(sys.stderr, level="DEBUG")

# Define conversation modes with their respective prompt templates
game_prompt = """You are the AI host and player for a game of Word Wrangler.

GAME RULES:
1. The user will be given a word or phrase that they must describe to you
2. The user CANNOT say any part of the word/phrase directly
3. You must try to guess the word/phrase based on the user's description
4. Once you guess correctly, the user will move on to their next word
5. The user is trying to get through as many words as possible in 60 seconds
6. The external application will handle timing and keeping score

YOUR ROLE:
1. Start with this exact brief introduction: "Welcome to Word Wrangler! I'll try to guess the words you describe. Remember, don't say any part of the word itself. Ready? Let's go!"
2. Listen carefully to the user's descriptions
3. Make intelligent guesses based on what they say
4. When you think you know the answer, state it clearly: "Is it [your guess]?"
5. If you're struggling, ask for more specific clues
6. Keep the game moving quickly - make guesses promptly
7. Be enthusiastic and encouraging

IMPORTANT:
- Keep all responses brief - the game is timed!
- Make multiple guesses if needed
- Use your common knowledge to make educated guesses
- If the user indicates you got it right, just say "Got it!" and prepare for the next word
- If you've made several wrong guesses, simply ask for "Another clue please?"

Start with the exact introduction specified above, then wait for the user to begin describing their first word."""

# Define personality presets
PERSONALITY_PRESETS = {
    "friendly": "You have a warm, approachable personality. You use conversational language, occasional humor, and express enthusiasm for the topic. Make the user feel comfortable and engaged.",
    "professional": "You have a formal, precise personality. You communicate clearly and directly with a focus on accuracy and relevance. Your tone is respectful and business-like.",
    "enthusiastic": "You have an energetic, passionate personality. You express excitement about the topic and use dynamic language. You're encouraging and positive throughout the conversation.",
    "thoughtful": "You have a reflective, philosophical personality. You speak carefully, considering multiple angles of each point. You ask thought-provoking questions and acknowledge nuance.",
    "witty": "You have a clever, humorous personality. While remaining informative, you inject appropriate wit and playful language. Your goal is to be engaging and entertaining while still being helpful.",
}


async def main(transport: DailyTransport, config: Dict[str, Any]):
    # Use the provided session logger if available, otherwise use the default logger
    logger.debug("Configuration: {}", config)

    # Extract configuration parameters with defaults
    personality = config.get("personality", "witty")

    personality_prompt = PERSONALITY_PRESETS.get(personality, PERSONALITY_PRESETS["friendly"])

    system_instruction = f"""{game_prompt}

{personality_prompt}

Important guidelines:
1. Your responses will be converted to speech, so keep them concise and conversational.
2. Don't use special characters or formatting that wouldn't be natural in speech.
3. Encourage the user to elaborate when appropriate."""

    intro_message = """Start with this exact brief introduction: "Welcome to Word Wrangler! I'll try to guess the words you describe. Remember, don't say any part of the word itself. Ready? Let's go!"""

    # Create the STT mute filter if we have strategies to apply
    stt_mute_filter = STTMuteFilter(
        config=STTMuteConfig(strategies={STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE})
    )

    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        transcribe_user_audio=True,
        system_instruction=system_instruction,
    )

    # Set up the initial context for the conversation
    messages = [
        {
            "role": "user",
            "content": intro_message,
        },
    ]

    # This sets up the LLM context by providing messages and tools
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt_mute_filter,
            context_aggregator.user(),
            llm,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.debug("Client ready event received")
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info("First participant joined: {}", participant["id"])
        # Capture the participant's transcription
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info("Participant left: {}", participant)
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)


async def bot(args: DailySessionArguments):
    """Main bot entry point compatible with the FastAPI route handler.

    Args:
        room_url: The Daily room URL
        token: The Daily room token
        body: The configuration object from the request body
        session_id: The session ID for logging
    """
    from pipecat.audio.filters.krisp_filter import KrispFilter

    logger.info(f"Bot process initialized {args.room_url} {args.token}")

    transport = DailyTransport(
        args.room_url,
        args.token,
        "Word Wrangler Bot",
        DailyParams(
            audio_in_enabled=True,
            audio_in_filter=None if LOCAL_RUN else KrispFilter(),
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    try:
        await main(transport, args.body)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise


# Local development
async def local_daily(args: DailySessionArguments):
    """Daily transport for local development."""
    # from runner import configure

    try:
        async with aiohttp.ClientSession() as session:
            transport = DailyTransport(
                room_url=args.room_url,
                token=args.token,
                bot_name="Bot",
                params=DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(),
                ),
            )

            test_config = {
                "personality": args.personality,
            }

            await main(transport, test_config)
    except Exception as e:
        logger.exception(f"Error in local development mode: {e}")


# Local development entry point
if LOCAL_RUN and __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Word Wrangler bot in local development mode"
    )
    parser.add_argument(
        "-u", "--room-url", type=str, default=os.getenv("DAILY_SAMPLE_ROOM_URL", "")
    )
    parser.add_argument(
        "-t", "--token", type=str, default=os.getenv("DAILY_SAMPLE_ROOM_TOKEN", None)
    )
    parser.add_argument(
        "-p",
        "--personality",
        default="witty",
        choices=["friendly", "professional", "enthusiastic", "thoughtful", "witty"],
        help="Personality preset for the bot (friendly, professional, enthusiastic, thoughtful, witty)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(local_daily(args))
    except Exception as e:
        logger.exception(f"Failed to run in local mode: {e}")
