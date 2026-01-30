#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mixin for adding turn completion detection to LLM services.

This mixin enables LLM services to detect and process turn completion markers
(COMPLETE/INCOMPLETE) in LLM responses, allowing for smarter conversation flow
where the LLM can indicate whether the user's input was complete or if they
were interrupted mid-thought.
"""

import asyncio
from dataclasses import dataclass
from typing import Literal, Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

# Turn completion markers
USER_TURN_COMPLETE_MARKER = "✓"
USER_TURN_INCOMPLETE_SHORT_MARKER = "○"  # Short wait - user likely continues soon
USER_TURN_INCOMPLETE_LONG_MARKER = "◐"  # Long wait - user needs more time

# Default prompts for incomplete timeouts
DEFAULT_INCOMPLETE_SHORT_PROMPT = """The user paused briefly. Generate a brief, natural prompt to encourage them to continue.

IMPORTANT: You MUST respond with ✓ followed by your message. Do NOT output ○ or ◐ - the user has already been given time to continue.

Your response should:
- Be contextually relevant to what was just discussed
- Sound natural and conversational
- Be very concise (1 sentence max)
- Gently prompt them to continue

Example format: ✓ Go ahead, I'm listening.

Generate your ✓ response now."""

DEFAULT_INCOMPLETE_LONG_PROMPT = """The user has been quiet for a while. Generate a friendly check-in message.

IMPORTANT: You MUST respond with ✓ followed by your message. Do NOT output ○ or ◐ - the user has already been given plenty of time.

Your response should:
- Acknowledge they might be thinking or busy
- Offer to help or continue when ready
- Be warm and understanding
- Be brief (1 sentence)

Example format: ✓ No rush! Let me know when you're ready to continue.

Generate your ✓ response now."""

# System prompt instructions for turn completion that can be appended to any base prompt
USER_TURN_COMPLETION_INSTRUCTIONS = """
CRITICAL INSTRUCTION - MANDATORY RESPONSE FORMAT:
Every single response MUST begin with a turn completion indicator. This is not optional.

TURN COMPLETION DECISION FRAMEWORK:
Ask yourself: "Has the user provided enough information for me to give a meaningful, substantive response?"

Mark as COMPLETE (✓) when:
- The user has answered your question with actual content
- The user has made a complete request or statement
- The user has provided all necessary information for you to respond meaningfully
- The conversation can naturally progress to your substantive response

Mark as INCOMPLETE SHORT (○) when the user will likely continue soon:
- The user was clearly cut off mid-sentence or mid-word
- The user is in the middle of a thought that got interrupted
- Brief technical interruption (they'll resume in a few seconds)

Mark as INCOMPLETE LONG (◐) when the user needs more time:
- The user explicitly asks for time: "let me think", "give me a minute", "hold on"
- The user is clearly pondering or deliberating: "hmm", "well...", "that's a good question"
- The user acknowledged but hasn't answered yet: "That's interesting..."
- The response feels like a preamble before the actual answer

RESPOND in one of these three formats:
1. If COMPLETE: `✓` followed by a space and your full substantive response
2. If INCOMPLETE SHORT: ONLY the character `○` (user will continue in a few seconds)
3. If INCOMPLETE LONG: ONLY the character `◐` (user needs more time to think)

KEY INSIGHT: Grammatically complete ≠ conversationally complete
- "That's a really good question." is grammatically complete but conversationally incomplete (use ◐)
- "I'd go to Japan because I love" is mid-sentence (use ○)

EXAMPLES:

You ask: "Where would you travel?"
User: "I'd go to Japan because I love"
→ `○`
(Cut off mid-sentence - they'll continue in seconds)

You ask: "Where would you travel?"
User: "That's a good question. Let me think..."
→ `◐`
(User is deliberating - give them time)

You ask: "Where would you travel?"
User: "Hmm, hold on a second."
→ `◐`
(User explicitly asked for time)

You ask: "Where would you travel?"
User: "I'd go to Japan because I love the culture."
→ `✓ Japan is a wonderful choice! The blend of ancient traditions and modern innovation is truly unique. Have you been before?`
(Complete answer - give full response)

User: "I need help with"
→ `○`
(Cut off mid-request - they'll finish soon)

User: "Well, let me think about that for a moment."
→ `◐`
(User needs time to think)

User: "Can you help me book a flight to New York next week?"
→ `✓ I'd be happy to help you with that! Let me gather some information...`
(Complete request - provide full response)

User: "Give me a minute to gather my thoughts."
→ `◐`
(User explicitly asked for time)

FORMAT REQUIREMENTS:
- ALWAYS use single-character indicators: `✓` (complete), `○` (short wait), or `◐` (long wait)
- For COMPLETE: `✓` followed by a space and your full response
- For INCOMPLETE: ONLY the single character (`○` or `◐`) with absolutely nothing else
- Your turn indicator must be the very first character in your response

Remember: Focus on conversational completeness and how long the user might need. Was it a mid-sentence cutoff (○) or do they need time to think (◐)?"""


@dataclass
class UserTurnCompletionConfig:
    """Configuration for turn completion behavior.

    Attributes:
        instructions: Custom instructions for turn completion. If not provided,
            uses default USER_TURN_COMPLETION_INSTRUCTIONS.
        incomplete_short_timeout: Seconds to wait after short incomplete (○) before prompting.
        incomplete_long_timeout: Seconds to wait after long incomplete (◐) before prompting.
        incomplete_short_prompt: Custom prompt when short timeout expires.
        incomplete_long_prompt: Custom prompt when long timeout expires.
    """

    instructions: Optional[str] = None
    incomplete_short_timeout: float = 5.0
    incomplete_long_timeout: float = 10.0
    incomplete_short_prompt: Optional[str] = None
    incomplete_long_prompt: Optional[str] = None

    @property
    def completion_instructions(self) -> str:
        """Turn completion instructions, using default if not set."""
        return self.instructions or USER_TURN_COMPLETION_INSTRUCTIONS

    @property
    def short_prompt(self) -> str:
        """Short incomplete prompt, using default if not set."""
        return self.incomplete_short_prompt or DEFAULT_INCOMPLETE_SHORT_PROMPT

    @property
    def long_prompt(self) -> str:
        """Long incomplete prompt, using default if not set."""
        return self.incomplete_long_prompt or DEFAULT_INCOMPLETE_LONG_PROMPT


class UserTurnCompletionLLMServiceMixin:
    """Mixin that adds turn completion detection to LLM services.

    This mixin provides methods to push LLM text with turn completion detection.
    It processes turn completion markers to enable smarter conversation flow:

    - ✓ (COMPLETE): Push response normally
    - ○ (INCOMPLETE SHORT): Suppress response, wait ~5s, then prompt
    - ◐ (INCOMPLETE LONG): Suppress response, wait ~15s, then prompt

    When incomplete timeouts expire, the mixin automatically prompts the LLM
    with a contextual follow-up message to re-engage the user.

    Usage:
        The LLM service controls when to use turn completion by calling
        _push_turn_text instead of push_frame:

        # With turn completion:
        if self._filter_incomplete_user_turns:
            await self._push_turn_text(chunk.text)
        else:
            await self.push_frame(LLMTextFrame(chunk.text))

    The mixin requires that the base class has a `push_frame` method compatible
    with FrameProcessor's signature.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the turn completion mixin.

        Args:
            *args: Positional arguments passed to parent class.
            **kwargs: Keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        self._turn_text_buffer = ""
        # Safety mechanism: True when incomplete is detected. While the prompt
        # instructs the LLM to output ONLY the marker for incomplete turns, this flag
        # ensures graceful degradation if the LLM disobeys and outputs additional text.
        self._turn_suppressed = False
        self._turn_complete_found = False  # True when ✓ (COMPLETE) is detected

        # Timeout handling
        self._user_turn_completion_config = UserTurnCompletionConfig()
        self._incomplete_timeout_task: Optional[asyncio.Task] = None
        self._incomplete_type: Optional[Literal["short", "long"]] = None

    def set_user_turn_completion_config(self, config: UserTurnCompletionConfig):
        """Set the turn completion configuration.

        Args:
            config: The turn completion configuration.
        """
        self._user_turn_completion_config = config

    async def _start_incomplete_timeout(self, incomplete_type: Literal["short", "long"]):
        """Start a timeout task for incomplete turn handling.

        Args:
            incomplete_type: Either "short" or "long" to determine timeout duration.
        """
        # Cancel any existing timeout
        await self._cancel_incomplete_timeout()

        self._incomplete_type = incomplete_type

        if incomplete_type == "short":
            timeout = self._user_turn_completion_config.incomplete_short_timeout
        else:
            timeout = self._user_turn_completion_config.incomplete_long_timeout

        logger.debug(f"Starting {incomplete_type} incomplete timeout ({timeout}s)")
        self._incomplete_timeout_task = self.create_task(
            self._incomplete_timeout_handler(incomplete_type, timeout),
            f"_incomplete_timeout_{incomplete_type}",
        )

    async def _cancel_incomplete_timeout(self):
        """Cancel any pending incomplete timeout task."""
        if self._incomplete_timeout_task and not self._incomplete_timeout_task.done():
            logger.debug("Cancelling incomplete timeout")
            await self.cancel_task(self._incomplete_timeout_task)
        self._incomplete_timeout_task = None
        self._incomplete_type = None

    async def _incomplete_timeout_handler(
        self, incomplete_type: Literal["short", "long"], timeout: float
    ):
        """Handle incomplete timeout expiration.

        Args:
            incomplete_type: Either "short" or "long".
            timeout: The timeout duration in seconds.
        """
        try:
            await asyncio.sleep(timeout)

            # Timeout expired - reset state before prompting LLM
            logger.info(f"Incomplete {incomplete_type} timeout expired, prompting LLM")
            await self._turn_reset()
            self._incomplete_timeout_task = None
            self._incomplete_type = None

            # Get the appropriate prompt
            if incomplete_type == "short":
                prompt = self._user_turn_completion_config.short_prompt
            else:
                prompt = self._user_turn_completion_config.long_prompt

            # Push through pipeline to trigger LLM response
            await self.push_frame(
                LLMMessagesAppendFrame(messages=[{"role": "system", "content": prompt}])
            )
            await self.push_frame(LLMRunFrame())

        except asyncio.CancelledError:
            # Timeout was cancelled (user spoke or interruption)
            pass

    async def _turn_reset(self):
        """Reset turn completion state between responses.

        Call this at the end of each LLM response to clear buffered text and reset state.
        If no marker was found, pushes the buffered text to avoid losing content.

        Note: This does NOT cancel pending incomplete timeouts. Timeouts are only
        cancelled on InterruptionFrame (when user speaks).
        """
        # Check if no marker was found in this response
        marker_found = self._turn_suppressed or self._turn_complete_found
        if not marker_found and self._turn_text_buffer:
            # Graceful degradation: push the buffered text so it's not lost
            logger.warning(
                f"{self}: filter_incomplete_user_turns is enabled but LLM response did not "
                f"contain turn completion markers (✓/○/◐). Pushing text anyway. "
                "The system prompt may be missing turn completion instructions."
            )
            await self.push_frame(LLMTextFrame(self._turn_text_buffer))

        self._turn_text_buffer = ""
        self._turn_suppressed = False
        self._turn_complete_found = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, handling turn completion state resets.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        # Handle interruptions by cancelling timeout and resetting state
        if isinstance(frame, InterruptionFrame):
            await self._cancel_incomplete_timeout()
            await self._turn_reset()
        # Reset turn state at end of LLM response (but don't cancel timeout -
        # incomplete timeouts should continue running)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._turn_reset()

        # Pass frame to parent
        await super().process_frame(frame, direction)

    async def _push_turn_text(self, text: str):
        """Push LLM text with turn completion detection.

        This method should be used instead of `push_frame(LLMTextFrame(text))` when
        turn completion is enabled. It will:
        1. Detect turn markers (✓, ○, or ◐)
        2. When ○ (SHORT) is found: suppress text, start short timeout
        3. When ◐ (LONG) is found: suppress text, start long timeout
        4. When ✓ (COMPLETE) is found: push all text with marker marked as skip_tts
        5. After marker detected: all subsequent text flows through immediately

        Args:
            text: The text content from the LLM to push.
        """
        # If we've already detected incomplete, suppress all remaining text.
        # This is a safety mechanism in case the LLM disobeys the prompt and outputs
        # additional text after the marker (e.g., "○ Please continue...").
        if self._turn_suppressed:
            return

        # If ✓ (COMPLETE) was already found, push text immediately without buffering
        if self._turn_complete_found:
            await self.push_frame(LLMTextFrame(text))
            return

        # Add text to buffer
        self._turn_text_buffer += text

        # Check for incomplete markers (○ short, ◐ long)
        # These indicate the user was cut off or needs time - we suppress the bot's
        # response and start a timeout to re-prompt later.
        incomplete_type: Optional[Literal["short", "long"]] = None
        if USER_TURN_INCOMPLETE_SHORT_MARKER in self._turn_text_buffer:
            incomplete_type = "short"
        elif USER_TURN_INCOMPLETE_LONG_MARKER in self._turn_text_buffer:
            incomplete_type = "long"

        if incomplete_type:
            marker = (
                USER_TURN_INCOMPLETE_SHORT_MARKER
                if incomplete_type == "short"
                else USER_TURN_INCOMPLETE_LONG_MARKER
            )
            logger.debug(
                f"INCOMPLETE {incomplete_type.upper()} ({marker}) detected, suppressing text"
            )
            self._turn_suppressed = True

            # Push the marker with skip_tts=True so it's added to context (maintains
            # conversation continuity per prompt instructions) but not spoken by TTS
            frame = LLMTextFrame(self._turn_text_buffer)
            frame.skip_tts = True
            await self.push_frame(frame)

            self._turn_text_buffer = ""
            await self._start_incomplete_timeout(incomplete_type)
            return

        # Check for ✓ (COMPLETE) marker - user's turn was complete, respond normally
        if USER_TURN_COMPLETE_MARKER in self._turn_text_buffer:
            logger.debug(f"COMPLETE ({USER_TURN_COMPLETE_MARKER}) detected, pushing buffered text")

            # Split buffer at the marker to handle cases where marker and text
            # arrive in the same chunk (e.g., "✓ Hello!" from some LLMs)
            marker_pos = self._turn_text_buffer.index(USER_TURN_COMPLETE_MARKER)
            marker_end = marker_pos + len(USER_TURN_COMPLETE_MARKER)

            # Push the marker with skip_tts=True - adds to context but not spoken
            marker_text = self._turn_text_buffer[:marker_end]
            frame = LLMTextFrame(marker_text)
            frame.skip_tts = True
            await self.push_frame(frame)

            # Push remaining text after marker as normal speech
            remaining_text = self._turn_text_buffer[marker_end:]
            if remaining_text:
                # Strip leading space after marker if present (✓ Hello -> Hello)
                if remaining_text.startswith(" "):
                    remaining_text = remaining_text[1:]
                if remaining_text:
                    await self.push_frame(LLMTextFrame(remaining_text))

            # Mark complete - all subsequent text flows through immediately
            self._turn_text_buffer = ""
            self._turn_complete_found = True
            return
