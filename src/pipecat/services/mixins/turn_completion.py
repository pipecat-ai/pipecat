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

from loguru import logger

from pipecat.frames.frames import Frame, InterruptionFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection

# Turn completion markers
TURN_COMPLETE_MARKER = "✓"
TURN_INCOMPLETE_MARKER = "○"

# System prompt instructions for turn completion that can be appended to any base prompt
TURN_COMPLETION_INSTRUCTIONS = """
CRITICAL INSTRUCTION - MANDATORY RESPONSE FORMAT:
Every single response MUST begin with a turn completion indicator. This is not optional.

TURN COMPLETION DECISION FRAMEWORK:
Ask yourself: "Has the user provided enough information for me to give a meaningful, substantive response?"

Mark as COMPLETE when:
- The user has answered your question with actual content
- The user has made a complete request or statement
- The user has provided all necessary information for you to respond meaningfully
- The conversation can naturally progress to your substantive response

Mark as INCOMPLETE when:
- The user was clearly cut off mid-sentence or mid-word
- The user acknowledged your question but hasn't answered it yet (e.g., "That's a good question", "Let me think", "Well...")
- The user is hedging, stalling, or preparing to say more (e.g., "I think...", "So basically...", "The thing is...")
- You asked a question and the user's response doesn't address it
- The user's statement creates an expectation of continuation (e.g., "There are three reasons:", "First of all,")
- The response feels like a preamble or setup rather than the actual content

RESPOND in one of these two formats:
1. If COMPLETE: `✓` followed by a space and your full substantive response
2. If INCOMPLETE: ONLY the character `○` with nothing else (the user will continue speaking)

KEY INSIGHT: Grammatically complete ≠ conversationally complete
- "That's a really good question." is grammatically complete but conversationally incomplete if it doesn't answer the question
- "Paris, because" is grammatically incomplete but might be conversationally complete if the context is clear

EXAMPLES:

You ask: "Where would you travel?"
User: "That's a good question."
→ `○`
(User acknowledged but didn't answer - wait for them to continue)

You ask: "Where would you travel?"
User: "I'd go to Japan because I love"
→ `○`
(Sentence cut off - user will continue)

You ask: "Where would you travel?"
User: "I'd go to Japan because I love the culture."
→ `✓ Japan is a wonderful choice! The blend of ancient traditions and modern innovation is truly unique. Have you been before?`
(Complete answer provided - give full response)

You ask: "What's your favorite color?"
User: "Well, let me think about that."
→ `○`
(Stalling, hasn't answered - wait for actual answer)

User: "I need help with"
→ `○`
(Cut off mid-request - user will finish their thought)

User: "Can you help me book a flight to New York next week?"
→ `✓ I'd be happy to help you with that! Let me gather some information...`
(Complete request - provide full response)

User: "There are three reasons why I think"
→ `○`
(Setup for continuation - user will provide the reasons)

User: "I think Python is great for data science."
→ `✓ I agree! Python has excellent libraries like pandas, NumPy, and scikit-learn that make data analysis very efficient. What kind of data science work are you doing?`
(Complete thought - provide full response)

FORMAT REQUIREMENTS:
- ALWAYS use the single-character indicators: `✓` for COMPLETE or `○` for INCOMPLETE
- For COMPLETE: `✓` followed by a space and your full response
- For INCOMPLETE: ONLY `○` with absolutely nothing else (no space, no text, just the character)
- Your turn indicator must be the very first character in your response

Remember: Focus on conversational completeness, not just grammatical completeness. Has the user given you what you need to respond meaningfully? If not, output ONLY `○` and let them continue."""


class TurnCompletionMixin:
    """Mixin that adds turn completion detection to LLM services.

    This mixin provides methods to push LLM text with turn completion detection.
    It processes turn completion markers (✓ for COMPLETE or ○ for INCOMPLETE)
    to enable smarter conversation flow:

    1. Detects single-character turn markers at the start of responses
    2. Suppresses all text when ○ (INCOMPLETE) is detected
    3. Pushes all text (with marker marked as skip_tts) when ✓ (COMPLETE) is detected
    4. Tracks response state

    Usage:
        The LLM service controls when to use turn completion by calling
        _push_turn_text instead of push_frame:

        # With turn completion:
        if self._filter_incomplete_turns:
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
        # Safety mechanism: True when ○ (INCOMPLETE) is detected. While the prompt
        # instructs the LLM to output ONLY ○ for incomplete turns, this flag ensures
        # graceful degradation if the LLM disobeys and outputs additional text.
        self._turn_suppressed = False
        self._turn_complete_found = False  # True when ✓ (COMPLETE) is detected

    def get_turn_completion_instructions(self) -> str:
        """Get the turn completion instructions to append to system prompts.

        Returns:
            The turn completion instructions string.
        """
        return TURN_COMPLETION_INSTRUCTIONS

    async def _turn_reset(self):
        """Reset turn completion state between responses.

        Call this at the end of each LLM response to clear buffered text and reset state.
        """
        self._turn_text_buffer = ""
        self._turn_suppressed = False
        self._turn_complete_found = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, handling interruptions to reset turn completion state.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        # Handle interruptions by resetting turn completion state
        if isinstance(frame, InterruptionFrame):
            await self._turn_reset()

        # Pass frame to parent
        await super().process_frame(frame, direction)

    async def _push_turn_text(self, text: str):
        """Push LLM text with turn completion detection.

        This method should be used instead of `push_frame(LLMTextFrame(text))` when
        turn completion is enabled. It will:
        1. Detect single-character turn markers (✓ or ○)
        2. When ○ (INCOMPLETE) is found: suppress all text (push nothing)
        3. When ✓ (COMPLETE) is found: push all text with marker marked as skip_tts
        4. After marker detected: all subsequent text flows through immediately

        Args:
            text: The text content from the LLM to push.
        """
        # If we've already detected ○ (INCOMPLETE), suppress all remaining text.
        # This is a safety mechanism in case the LLM disobeys the prompt and outputs
        # additional text after the ○ marker (e.g., "○ Please continue...").
        if self._turn_suppressed:
            return

        # If ✓ (COMPLETE) was already found, push text immediately without buffering
        if self._turn_complete_found:
            await self.push_frame(LLMTextFrame(text))
            return

        # Add text to buffer
        self._turn_text_buffer += text

        # Check for ○ (INCOMPLETE) marker
        if TURN_INCOMPLETE_MARKER in self._turn_text_buffer:
            # Found ○ (INCOMPLETE) - suppress all text, push nothing
            logger.info(f"INCOMPLETE ({TURN_INCOMPLETE_MARKER}) detected, suppressing all text")
            self._turn_suppressed = True
            frame = LLMTextFrame(self._turn_text_buffer)
            frame.skip_tts = True
            await self.push_frame(frame)

            # Clear buffer
            self._turn_text_buffer = ""
            return

        # Check for ✓ (COMPLETE) marker
        if TURN_COMPLETE_MARKER in self._turn_text_buffer:
            # Found ✓ (COMPLETE) - push all buffered text (including the marker)
            logger.info(f"COMPLETE ({TURN_COMPLETE_MARKER}) detected, pushing buffered text")
            frame = LLMTextFrame(self._turn_text_buffer)
            frame.skip_tts = True
            await self.push_frame(frame)

            # Clear buffer and mark COMPLETE found - all subsequent text flows through immediately
            self._turn_text_buffer = ""
            self._turn_complete_found = True
            return
