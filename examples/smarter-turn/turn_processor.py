#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from loguru import logger

from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

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


class TurnProcessor(FrameProcessor):
    """Processes turn completion status from LLM responses.

    This processor monitors AggregatedTextFrame messages for COMPLETE/INCOMPLETE
    status indicators and manages the flow of text frames accordingly. When a
    response is marked as INCOMPLETE, it suppresses subsequent text frames until
    a COMPLETE status is received.
    """

    def __init__(self):
        self._processing_response = False
        super().__init__()

    async def _handle_aggregated_text(self, frame: AggregatedTextFrame):
        # if the frame.text is "INCOMPLETE", then we don't push the following
        # AggregatedTextFrame for the remainder of the response.
        if not self._processing_response:
            return

        # Regardless of the turn status, we push a <turn>STATUS</turn> frame to ensure the LLM
        # sees the context following the ruleset from the system prompt.
        # The frames pushed are marked as skip_tts to avoid TTS processing.
        # All other AggregatedTextFrames are pushed downstream, as long as we're processing a
        # response (e.g. the status is COMPLETE).
        if frame.text == "INCOMPLETE":
            logger.info("INCOMPLETE response detected, waiting for complete response")
            self._processing_response = False
            text_frame = TextFrame(text=f"<turn>INCOMPLETE</turn>")
            text_frame.skip_tts = True
            await self.push_frame(text_frame)
        elif frame.text == "COMPLETE":
            logger.info("COMPLETE response detected, pushing text frame")
            text_frame = TextFrame(text=f"<turn>COMPLETE</turn>")
            text_frame.skip_tts = True
            await self.push_frame(text_frame)
        else:
            await self.push_frame(frame)

    async def _handle_llm_start(self, _: LLMFullResponseStartFrame):
        self._processing_response = True

    async def _handle_llm_end(self, _: LLMFullResponseEndFrame):
        self._processing_response = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            await self._handle_llm_start(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, AggregatedTextFrame):
            await self._handle_aggregated_text(frame)
        else:
            await self.push_frame(frame, direction)
