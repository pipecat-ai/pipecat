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

# System prompt instructions for turn completion that can be appended to any base prompt
TURN_COMPLETION_INSTRUCTIONS = """
CRITICAL INSTRUCTION - MANDATORY RESPONSE FORMAT:
Every single response MUST begin with a turn completion tag. This is not optional.

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
1. `<turn>COMPLETE</turn>` followed by your substantive response
2. `<turn>INCOMPLETE</turn>` followed by a brief prompt to continue (e.g., "Please go on", "I'm listening")

KEY INSIGHT: Grammatically complete ≠ conversationally complete
- "That's a really good question." is grammatically complete but conversationally incomplete if it doesn't answer the question
- "Paris, because" is grammatically incomplete but might be conversationally complete if the context is clear

EXAMPLES:

You ask: "Where would you travel?"
User: "That's a good question."
→ `<turn>INCOMPLETE</turn> Take your time, I'm curious to hear your answer.`
(User acknowledged but didn't answer)

You ask: "Where would you travel?"
User: "I'd go to Japan because I love"
→ `<turn>INCOMPLETE</turn> Please continue, I'd love to hear more.`
(Sentence cut off)

You ask: "Where would you travel?"
User: "I'd go to Japan because I love the culture."
→ `<turn>COMPLETE</turn> Japan is a wonderful choice! The blend of ancient traditions and modern innovation is truly unique. Have you been before?`
(Complete answer provided)

You ask: "What's your favorite color?"
User: "Well, let me think about that."
→ `<turn>INCOMPLETE</turn> Take your time.`
(Stalling, hasn't answered)

User: "I need help with"
→ `<turn>INCOMPLETE</turn> Yes, what do you need help with?`
(Cut off mid-request)

User: "Can you help me book a flight to New York next week?"
→ `<turn>COMPLETE</turn> I'd be happy to help you with that! Let me gather some information...`
(Complete request)

FORMAT REQUIREMENTS:
- ALWAYS use full XML tags: `<turn>COMPLETE</turn>` or `<turn>INCOMPLETE</turn>`
- NEVER respond with just "COMPLETE" or "INCOMPLETE" without tags
- Your turn tag must be the very first thing in your response

Remember: Focus on conversational completeness, not just grammatical completeness. Has the user given you what you need to respond meaningfully?"""


class TurnCompletionMixin:
    """Mixin that adds turn completion detection to LLM services.

    This mixin provides methods to push LLM text with turn completion detection.
    It processes turn completion markers (<turn>COMPLETE</turn> or
    <turn>INCOMPLETE</turn>) to enable smarter conversation flow:

    1. Aggregates text looking for turn completion tags
    2. Pushes AggregatedTextFrame with the status (COMPLETE/INCOMPLETE)
    3. Modifies TextFrames to mark turn status frames as skip_tts
    4. Tracks response state

    Usage:
        The LLM service controls when to use turn completion by calling
        the appropriate methods:

        # With turn completion:
        await self._turn_start_response()
        await self._push_turn_text(chunk.text)
        await self._turn_end_response()

        # Without turn completion:
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
        self._turn_suppressed = False  # True when INCOMPLETE is detected
        self._turn_complete_found = False  # True when COMPLETE is detected

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

    def _is_building_turn_marker(self, buffer: str) -> bool:
        """Check if the buffer is potentially building a turn marker.

        Checks progressively: <, <t, <tu, <tur, <turn, <turn>, <turn>C, etc.

        Args:
            buffer: The current text buffer.

        Returns:
            True if we might be building a marker, False otherwise.
        """
        # Check if we're building the opening tag: <turn>
        opening_tag = "<turn>"
        for i in range(1, len(opening_tag) + 1):
            if buffer.endswith(opening_tag[:i]):
                return True

        # Check if we have opening tag and might be building content or closing tag
        if "<turn>" in buffer:
            # We have the opening tag
            # Check if we're building the closing tag: </turn>
            closing_tag = "</turn>"
            for i in range(1, len(closing_tag) + 1):
                if buffer.endswith(closing_tag[:i]):
                    return True

            # Check if we're building COMPLETE or INCOMPLETE
            # We need to keep buffering until we see the closing tag
            if "</turn>" not in buffer:
                return True

        return False

    async def _push_turn_text(self, text: str):
        """Push LLM text with turn completion detection.

        This method should be used instead of `push_frame(LLMTextFrame(text))` when
        turn completion is enabled. It will:
        1. Buffer text and look for turn completion markers
        2. When INCOMPLETE is found: suppress all text (push nothing)
        3. When COMPLETE is found: push all buffered text, then push subsequent text immediately
        4. After COMPLETE: all subsequent text flows through immediately without buffering

        Args:
            text: The text content from the LLM to push.
        """
        # If we've already detected INCOMPLETE, suppress all remaining text
        if self._turn_suppressed:
            return

        # If COMPLETE was already found, push text immediately without buffering
        if self._turn_complete_found:
            await self.push_frame(LLMTextFrame(text))
            return

        # Add text to buffer
        self._turn_text_buffer += text

        # Check for INCOMPLETE marker
        if "<turn>INCOMPLETE</turn>" in self._turn_text_buffer:
            # Found INCOMPLETE - suppress all text, push nothing
            self._turn_suppressed = True
            logger.info("INCOMPLETE response detected, suppressing all text")
            self._turn_text_buffer = ""
            return

        # Check for COMPLETE marker
        if "<turn>COMPLETE</turn>" in self._turn_text_buffer:
            # Found COMPLETE - push all buffered text (including the marker)
            logger.info("COMPLETE response detected, pushing buffered text")
            frame = LLMTextFrame(self._turn_text_buffer)
            frame.skip_tts = True
            await self.push_frame(frame)

            # Clear buffer and mark COMPLETE found - all subsequent text flows through immediately
            self._turn_text_buffer = ""
            self._turn_complete_found = True
            return

        # Check if we're building a marker - if so, keep buffering
        if self._is_building_turn_marker(self._turn_text_buffer):
            return

        # Not building a marker - push the buffered text and clear
        if self._turn_text_buffer:
            await self.push_frame(LLMTextFrame(self._turn_text_buffer))
            self._turn_text_buffer = ""
