#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Word Wrangler: A voice-based word guessing game.

To run this demo:
1. Set up environment variables:
   - GOOGLE_API_KEY: API key for Google services
   - GOOGLE_TEST_CREDENTIALS_FILE: Path to Google credentials JSON file

2. Install requirements:
   pip install -r requirements.txt

3. Run in local development mode:
   LOCAL_RUN=1 python word_wrangler.py
"""

import asyncio
import os
import re
import sys
from typing import Any, Mapping, Optional

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecatcloud.agent import DailySessionArguments
from word_list import generate_game_words

from pipecat.audio.utils import create_default_resampler
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
)
from pipecat.processors.consumer_processor import ConsumerProcessor
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.producer_processor import ProducerProcessor
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
    GeminiMultimodalModalities,
    InputParams,
)
from pipecat.services.google.tts import GoogleTTSService
from pipecat.sync.base_notifier import BaseNotifier
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.utils.text.base_text_filter import BaseTextFilter

load_dotenv(override=True)

# Check if we're in local development mode
LOCAL_RUN = os.getenv("LOCAL_RUN")
if LOCAL_RUN:
    import webbrowser

    try:
        from runner import configure
    except ImportError:
        logger.error("Could not import local_runner module. Local development mode may not work.")


logger.add(sys.stderr, level="DEBUG")

GAME_DURATION_SECONDS = 120
NUM_WORDS_PER_GAME = 20
HOST_VOICE_ID = "en-US-Chirp3-HD-Charon"
PLAYER_VOICE_ID = "Kore"

# Define conversation modes with their respective prompt templates
game_player_prompt = """You are a player for a game of Word Wrangler.

GAME RULES:
1. The user will be given a word or phrase that they must describe to you
2. The user CANNOT say any part of the word/phrase directly
3. You must try to guess the word/phrase based on the user's description
4. Once you guess correctly, the user will move on to their next word
5. The user is trying to get through as many words as possible in 60 seconds
6. The external application will handle timing and keeping score

YOUR ROLE:
1. Listen carefully to the user's descriptions
2. Make intelligent guesses based on what they say
3. When you think you know the answer, state it clearly: "Is it [your guess]?"
4. If you're struggling, ask for more specific clues
5. Keep the game moving quickly - make guesses promptly
6. Be enthusiastic and encouraging

IMPORTANT:
- Keep all responses brief - the game is timed!
- Make multiple guesses if needed
- Use your common knowledge to make educated guesses
- If the user indicates you got it right, just say "Got it!" and prepare for the next word
- If you've made several wrong guesses, simply ask for "Another clue please?"

Start by guessing once you hear the user describe the word or phrase."""

game_host_prompt = """You are the AI host for a game of Word Wrangler. There are two players in the game: the human describer and the AI guesser.

GAME RULES:
1. You, the host, will give the human describer a word or phrase that they must describe
2. The describer CANNOT say any part of the word/phrase directly
3. The AI guesser will try to guess the word/phrase based on the describer's description
4. Once the guesser guesses correctly, move on to the next word
5. The describer is trying to get through as many words as possible in 60 seconds
6. The describer can say "skip" or "pass" to get a new word if they find a word too difficult
7. The describer can ask you to repeat the current word if they didn't hear it clearly
8. You'll keep track of the score (1 point for each correct guess)
9. The external application will handle timing

YOUR ROLE:
1. Start with this exact brief introduction: "Welcome to Word Wrangler! I'll give you words to describe, and the A.I. player will try to guess them. Remember, don't say any part of the word itself. Here's your first word: [word]."
2. Provide words to the describer. Choose 1 or 2 word phrases that cover a variety of topics, including animals, objects, places, and actions.
3. IMPORTANT: You will hear DIFFERENT types of input:
   a. DESCRIPTIONS from the human (which you should IGNORE)
   b. AFFIRMATIONS from the human (like "correct", "that's right", "you got it") which you should IGNORE
   c. GUESSES from the AI player (which will be in the form of "Is it [word]?" or similar question format)
   d. SKIP REQUESTS from the human (if they say "skip", "pass", or "next word please")
   e. REPEAT REQUESTS from the human (if they say "repeat", "what was that?", "say again", etc.)

4. HOW TO RESPOND:
   - If you hear a DESCRIPTION or AFFIRMATION from the human, respond with exactly "IGNORE" (no other text)
   - If you hear a GUESS (in question form) and it's INCORRECT, respond with exactly "NO" (no other text)
   - If you hear a GUESS (in question form) and it's CORRECT, respond with "Correct! That's [N] points. Your next word is [new word]" where N is the current score
   - If you hear a SKIP REQUEST, respond with "The new word is [new word]" (don't change the score)
   - If you hear a REPEAT REQUEST, respond with "Your word is [current word]" (don't change the score)

5. SCORING:
   - Start with a score of 0
   - Add 1 point for each correct guess by the AI player
   - Do NOT add points for skipped words
   - Announce the current score after every correct guess

RESPONSE EXAMPLES:
- Human says: "This is something you use to write" → You respond: "IGNORE"
- Human says: "That's right!" or "You got it!" → You respond: "IGNORE"
- Human says: "Wait, what was my word again?" → You respond: "Your word is [current word]"
- Human says: "Can you repeat that?" → You respond: "Your word is [current word]"
- AI says: "Is it a pen?" → If correct and it's the first point, you respond: "Correct! That's 1 point. Your next word is [new word]"
- AI says: "Is it a pencil?" → If correct and it's the third point, you respond: "Correct! That's 3 points. Your next word is [new word]"
- AI says: "Is it a marker?" → If incorrect, you respond: "NO"
- Human says: "Skip this one" or "Pass" → You respond: "The new word is [new word]"

IMPORTANT GUIDELINES:
- Choose words that range from easy to moderately difficult
- Keep all responses brief - the game is timed!
- Your "NO" and "IGNORE" responses won't be verbalized, but will be visible in the chat
- Always keep track of the CURRENT word so you can repeat it when asked
- Always keep track of the CURRENT SCORE and announce it after every correct guess
- Make sure your word choices are appropriate for all audiences
- If the human asks to skip, always provide a new word immediately without changing the score
- If the human asks you to repeat the word, say ONLY "Your word is [current word]" - don't add additional text
- CRUCIAL: Never interpret the human saying "correct", "that's right", "good job", or similar affirmations as a correct guess. These are just the human giving feedback to the AI player.

Start with the exact introduction specified above and give the first word."""


class HostResponseTextFilter(BaseTextFilter):
    """Custom text filter for Word Wrangler game.

    This filter removes "NO" and "IGNORE" responses from the host so they don't get verbalized,
    allowing for silent incorrect guess handling and ignoring descriptions.
    """

    def __init__(self):
        self._interrupted = False

    def update_settings(self, settings: Mapping[str, Any]):
        # No settings to update for this filter
        pass

    async def filter(self, text: str) -> str:
        # Remove case and whitespace for comparison
        clean_text = text.strip().upper()

        # If the text is exactly "NO" or "IGNORE", return empty string
        if clean_text == "NO" or clean_text == "IGNORE":
            return ""

        return text

    async def handle_interruption(self):
        self._interrupted = True

    async def reset_interruption(self):
        self._interrupted = False


class BotStoppedSpeakingNotifier(FrameProcessor):
    """A processor that notifies whenever a BotStoppedSpeakingFrame is detected."""

    def __init__(self, notifier: BaseNotifier):
        super().__init__()
        self._notifier = notifier

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Check if this is a BotStoppedSpeakingFrame
        if isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug(f"{self}: Host bot stopped speaking, notifying listeners")
            await self._notifier.notify()

        # Always push the frame through
        await self.push_frame(frame, direction)


class StartFrameGate(FrameProcessor):
    """A gate that blocks only StartFrame until notified by a notifier.

    Once opened, all frames pass through normally.
    """

    def __init__(self, notifier: BaseNotifier):
        super().__init__()
        self._notifier = notifier
        self._blocked_start_frame: Optional[Frame] = None
        self._gate_opened = False
        self._gate_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._gate_opened:
            # Once the gate is open, let everything through
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartFrame):
            # Store the StartFrame and wait for notification
            logger.debug(f"{self}: Blocking StartFrame until host bot stops speaking")
            self._blocked_start_frame = frame

            # Start the gate task if not already running
            if not self._gate_task:
                self._gate_task = self.create_task(self._wait_for_notification())

    async def _wait_for_notification(self):
        try:
            # Wait for the notifier
            await self._notifier.wait()

            # Gate is now open - only run this code once
            if not self._gate_opened:
                self._gate_opened = True
                logger.debug(f"{self}: Gate opened, passing through blocked StartFrame")

                # Push the blocked StartFrame if we have one
                if self._blocked_start_frame:
                    await self.push_frame(self._blocked_start_frame)
                    self._blocked_start_frame = None
        except asyncio.CancelledError:
            logger.debug(f"{self}: Gate task was cancelled")
            raise
        except Exception as e:
            logger.exception(f"{self}: Error in gate task: {e}")
            raise


class GameStateTracker(FrameProcessor):
    """Tracks game state including new words and score by monitoring host responses."""

    def __init__(self, new_word_notifier: BaseNotifier):
        super().__init__()
        self._new_word_notifier = new_word_notifier
        self._text_buffer = ""
        self._current_score = 0

        # Words/phrases that indicate a new word being provided
        self._key_phrases = ["your word is", "new word is", "next word is"]

        # Pattern to extract score from responses
        self._score_pattern = re.compile(r"that's (\d+) point", re.IGNORECASE)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Collect text from LLMTextFrames
        if isinstance(frame, LLMTextFrame):
            text = frame.text

            # Skip responses that are "NO" or "IGNORE"
            if text.strip() in ["NO", "IGNORE"]:
                logger.debug(f"Skipping NO/IGNORE response")
                await self.push_frame(frame, direction)
                return

            # Add the new text to our buffer
            self._text_buffer += text

        # Process complete responses when we get an end frame
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._text_buffer:
                buffer_lower = self._text_buffer.lower()

                # 1. Check for new word announcements
                new_word_detected = False
                for phrase in self._key_phrases:
                    if phrase in buffer_lower:
                        await self._new_word_notifier.notify()
                        new_word_detected = True
                        break

                if not new_word_detected:
                    logger.debug(f"No new word phrases detected")

                # 2. Check for score updates
                score_match = self._score_pattern.search(buffer_lower)
                if score_match:
                    try:
                        score = int(score_match.group(1))
                        # Only update if the new score is higher
                        if score > self._current_score:
                            logger.debug(f"Score updated from {self._current_score} to {score}")
                            self._current_score = score
                        else:
                            logger.debug(
                                f"Ignoring score {score} <= current score {self._current_score}"
                            )
                    except ValueError as e:
                        logger.warning(f"Error parsing score: {e}")
                else:
                    logger.debug(f"No score pattern match in: '{buffer_lower}'")

                # Reset the buffer after processing the complete response
                self._text_buffer = ""

        # Always push the frame through
        await self.push_frame(frame, direction)

    @property
    def current_score(self) -> int:
        """Get the current score."""
        return self._current_score


class GameTimer:
    """Manages the game timer and triggers end-game events."""

    def __init__(
        self,
        task: PipelineTask,
        game_state_tracker: GameStateTracker,
        game_duration_seconds: int = 120,
    ):
        self._task = task
        self._game_state_tracker = game_state_tracker
        self._game_duration = game_duration_seconds
        self._timer_task = None
        self._start_time = None

    def start(self):
        """Start the game timer."""
        if self._timer_task is None:
            self._start_time = asyncio.get_event_loop().time()
            self._timer_task = asyncio.create_task(self._run_timer())
            logger.info(f"Game timer started: {self._game_duration} seconds")

    def stop(self):
        """Stop the game timer."""
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None
            logger.info("Game timer stopped")

    def get_remaining_time(self) -> int:
        """Get the remaining time in seconds."""
        if self._start_time is None:
            return self._game_duration

        elapsed = asyncio.get_event_loop().time() - self._start_time
        remaining = max(0, self._game_duration - int(elapsed))
        return remaining

    async def _run_timer(self):
        """Run the timer and end the game when time is up."""
        try:
            # Wait for the game duration
            await asyncio.sleep(self._game_duration)

            # Game time is up, get the final score
            final_score = self._game_state_tracker.current_score

            # Create end game message
            end_message = f"Time's up! Thank you for playing Word Wrangler. Your final score is {final_score} point"
            if final_score != 1:
                end_message += "s"
            end_message += ". Great job!"

            # Send end game message as TTSSpeakFrame
            logger.info(f"Game over! Final score: {final_score}")
            await self._task.queue_frames([TTSSpeakFrame(text=end_message)])

            # End the game
            await self._task.queue_frames([EndFrame()])

        except asyncio.CancelledError:
            logger.debug("Game timer task cancelled")
        except Exception as e:
            logger.exception(f"Error in game timer: {e}")


class ResettablePlayerLLM(GeminiMultimodalLiveLLMService):
    """A specialized LLM service that can reset its context when notified about a new word.

    This LLM intelligently waits for the host to finish speaking before reconnecting.
    """

    def __init__(
        self,
        api_key: str,
        system_instruction: str,
        new_word_notifier: BaseNotifier,
        host_stopped_speaking_notifier: BaseNotifier,
        voice_id: str = PLAYER_VOICE_ID,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key, voice_id=voice_id, system_instruction=system_instruction, **kwargs
        )
        self._new_word_notifier = new_word_notifier
        self._host_stopped_speaking_notifier = host_stopped_speaking_notifier
        self._base_system_instruction = system_instruction
        self._reset_task: Optional[asyncio.Task] = None
        self._pending_reset: bool = False

    async def start(self, frame: StartFrame):
        await super().start(frame)

        # Start the notifier listener task
        if not self._reset_task or self._reset_task.done():
            self._reset_task = self.create_task(self._listen_for_notifications())

    async def stop(self, frame: EndFrame):
        # Cancel the reset task if it exists
        if self._reset_task and not self._reset_task.done():
            await self.cancel_task(self._reset_task)
            self._reset_task = None

        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        # Cancel the reset task if it exists
        if self._reset_task and not self._reset_task.done():
            await self.cancel_task(self._reset_task)
            self._reset_task = None

        await super().cancel(frame)

    async def _listen_for_notifications(self):
        """Listen for new word and host stopped speaking notifications."""
        try:
            # Create tasks for both notifiers
            new_word_task = self.create_task(self._listen_for_new_word())
            host_stopped_task = self.create_task(self._listen_for_host_stopped())

            # Wait for both tasks to complete (which should never happen)
            await asyncio.gather(new_word_task, host_stopped_task)

        except asyncio.CancelledError:
            logger.debug(f"{self}: Notification listener tasks cancelled")
            raise
        except Exception as e:
            logger.exception(f"{self}: Error in notification listeners: {e}")
            raise

    async def _listen_for_new_word(self):
        """Listen for new word notifications and flag a reset is needed."""
        while True:
            # Wait for a new word notification
            await self._new_word_notifier.wait()
            logger.info(
                f"{self}: Received new word notification, disconnecting and waiting for host to finish"
            )

            # Disconnect immediately to stop processing
            await self._disconnect()

            # Reset the system instruction
            self._system_instruction = self._base_system_instruction

            # Flag that we need to reconnect when the host stops speaking
            self._pending_reset = True

    async def _listen_for_host_stopped(self):
        """Listen for host stopped speaking and reconnect if a reset is pending."""
        while True:
            # Wait for host stopped speaking notification
            await self._host_stopped_speaking_notifier.wait()

            # If we have a pending reset, reconnect now
            if self._pending_reset:
                logger.info(f"{self}: Host finished speaking, completing the LLM reset")

                # Reconnect
                await self._connect()

                # Reset the flag
                self._pending_reset = False

                logger.info(f"{self}: LLM reset complete")


async def tts_audio_raw_frame_filter(frame: Frame):
    """Filter to check if the frame is a TTSAudioRawFrame."""
    return isinstance(frame, TTSAudioRawFrame)


# Create a resampler instance once
resampler = create_default_resampler()


async def tts_to_input_audio_transformer(frame: Frame):
    """Transform TTS audio frames to InputAudioRawFrame with resampling.

    Converts 24kHz TTS output to 16kHz input audio required by the player LLM.

    Args:
        frame (Frame): The frame to transform (expected to be TTSAudioRawFrame)

    Returns:
        InputAudioRawFrame: The transformed and resampled input audio frame
    """
    if isinstance(frame, TTSAudioRawFrame):
        # Resample the audio from 24kHz to 16kHz
        resampled_audio = await resampler.resample(
            frame.audio,
            frame.sample_rate,  # Source rate (24kHz)
            16000,  # Target rate (16kHz)
        )

        # Create a new InputAudioRawFrame with the resampled audio
        input_frame = InputAudioRawFrame(
            audio=resampled_audio,
            sample_rate=16000,  # New sample rate
            num_channels=frame.num_channels,
        )
        return input_frame


async def main(room_url: str, token: str):
    # Use the provided session logger if available, otherwise use the default logger
    logger.debug("Starting bot in room: {}", room_url)

    game_words = generate_game_words(NUM_WORDS_PER_GAME)
    words_string = ", ".join(f'"{word}"' for word in game_words)
    logger.debug(f"Game words: {words_string}")

    transport = DailyTransport(
        room_url,
        token,
        "Word Wrangler Bot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    player_instruction = f"""{game_player_prompt}

Important guidelines:
1. Your responses will be converted to speech, so keep them concise and conversational.
2. Don't use special characters or formatting that wouldn't be natural in speech.
3. Encourage the user to elaborate when appropriate."""

    host_instruction = f"""{game_host_prompt}

GAME WORDS:
Use ONLY these words for the game (in any order): {words_string}

Important guidelines:
1. Your responses will be converted to speech, so keep them concise and conversational.
2. Don't use special characters or formatting that wouldn't be natural in speech.
3. ONLY use words from the provided list above when giving words to the player."""

    intro_message = """Start with this exact brief introduction: "Welcome to Word Wrangler! I'll give you words to describe, and the A.I. player will try to guess them. Remember, don't say any part of the word itself. Here's your first word: [word]." """

    # Create the STT mute filter if we have strategies to apply
    stt_mute_filter = STTMuteFilter(
        config=STTMuteConfig(strategies={STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE})
    )

    host_llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=host_instruction,
        params=InputParams(modalities=GeminiMultimodalModalities.TEXT),
    )

    host_tts = GoogleTTSService(
        voice_id=HOST_VOICE_ID,
        credentials_path=os.getenv("GOOGLE_TEST_CREDENTIALS_FILE"),
        text_filters=[HostResponseTextFilter()],
    )

    producer = ProducerProcessor(
        filter=tts_audio_raw_frame_filter,
        transformer=tts_to_input_audio_transformer,
        passthrough=True,
    )
    consumer = ConsumerProcessor(producer=producer)

    # Create the notifiers
    bot_speaking_notifier = EventNotifier()
    new_word_notifier = EventNotifier()

    # Create BotStoppedSpeakingNotifier to detect when host bot stops speaking
    bot_stopped_speaking_detector = BotStoppedSpeakingNotifier(bot_speaking_notifier)

    # Create StartFrameGate to block Player LLM until host has stopped speaking
    start_frame_gate = StartFrameGate(bot_speaking_notifier)

    # Create GameStateTracker to handle new words and score tracking
    game_state_tracker = GameStateTracker(new_word_notifier)

    # Create a resettable player LLM that coordinates between notifiers
    player_llm = ResettablePlayerLLM(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=player_instruction,
        new_word_notifier=new_word_notifier,
        host_stopped_speaking_notifier=bot_speaking_notifier,
        voice_id=PLAYER_VOICE_ID,
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
    context_aggregator = host_llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Receive audio/video from Daily call
            stt_mute_filter,  # Filter out speech during the bot's initial turn
            ParallelPipeline(
                # Host branch: manages the game and provides words
                [
                    consumer,  # Receives audio from the player branch
                    host_llm,  # AI host that provides words and tracks score
                    game_state_tracker,  # Tracks words and score from host responses
                    host_tts,  # Converts host text to speech
                    bot_stopped_speaking_detector,  # Notifies when host stops speaking
                ],
                # Player branch: guesses words based on human descriptions
                [
                    start_frame_gate,  # Gates the player until host finishes intro
                    player_llm,  # AI player that makes guesses
                    producer,  # Collects audio frames to be passed to the consumer
                ],
            ),
            transport.output(),  # Send audio/video back to Daily call
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Create the game timer
    game_timer = GameTimer(task, game_state_tracker, game_duration_seconds=GAME_DURATION_SECONDS)

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info("First participant joined: {}", participant["id"])
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        # Start the game timer
        game_timer.start()

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info("Participant left: {}", participant)
        # Stop the timer
        game_timer.stop()
        # Cancel the pipeline task
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
    logger.info(f"Bot process initialized {args.room_url} {args.token}")

    try:
        await main(args.room_url, args.token)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise


# Local development functions
async def local_main():
    """Function for local development testing."""
    try:
        async with aiohttp.ClientSession() as session:
            (room_url, token) = await configure(session)
            logger.warning("_")
            logger.warning("_")
            logger.warning(f"Talk to your voice agent here: {room_url}")
            logger.warning("_")
            logger.warning("_")
            webbrowser.open(room_url)
            await main(room_url, token)
    except Exception as e:
        logger.exception(f"Error in local development mode: {e}")


# Local development entry point
if LOCAL_RUN and __name__ == "__main__":
    try:
        asyncio.run(local_main())
    except Exception as e:
        logger.exception(f"Failed to run in local mode: {e}")
