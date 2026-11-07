#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn completion markers, the instructions that ask for them, and their configuration.

An LLM starts each response with a turn completion marker (●, ◐, ○), which
:class:`~pipecat.services.llm_extensions.user_turn_completion.UserTurnCompletionLLMExtension` reads
and reports so a user turn stop strategy can decide whether the user was done.
"""

from dataclasses import dataclass
from enum import Enum

# Turn completion markers. Fill level tracks how much of the user's turn has
# arrived: full is a finished turn, half is a turn cut off mid-thought, empty is
# a user who hasn't started answering. Each is a single token in every major
# tokenizer, which matters because the complete marker is generated before any
# speakable text.
USER_TURN_COMPLETE_MARKER = "●"
USER_TURN_INCOMPLETE_SHORT_MARKER = "◐"  # Short wait - user likely continues soon
USER_TURN_INCOMPLETE_LONG_MARKER = "○"  # Long wait - user needs more time


class TurnMarker(Enum):
    """Completion verdict detected in the current LLM response.

    - ``COMPLETE``:   ● detected, response flows through as speech.
    - ``INCOMPLETE``: ◐/○ detected, the rest of the response is dropped.
    """

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class IncompleteType(Enum):
    """How long to wait before re-prompting after an incomplete turn.

    - ``SHORT``: ◐ detected, the user was cut off and likely continues soon.
    - ``LONG``:  ○ detected, the user needs more time to think.
    """

    SHORT = "short"
    LONG = "long"


# Default prompts for incomplete timeouts
def _render_incomplete_short_prompt(complete: str, short: str, long: str) -> str:
    """Build the re-prompt sent when a short incomplete-turn timeout expires.

    Args:
        complete: Marker the LLM emits when the user turn is complete.
        short: Marker for a turn cut off mid-thought.
        long: Marker for a user who needs more time.

    Returns:
        The prompt text with the markers substituted in.
    """
    return f"""The user paused briefly. Generate a brief, natural prompt to encourage them to continue.

IMPORTANT: You MUST respond with {complete} followed by your message. Do NOT output {short} or {long} - the user has already been given time to continue.

Your response should:
- Be contextually relevant to what was just discussed
- Sound natural and conversational
- Be very concise (1 sentence max)
- Gently prompt them to continue

Example format: {complete} Go ahead, I'm listening.

Generate your {complete} response now."""


DEFAULT_INCOMPLETE_SHORT_PROMPT = _render_incomplete_short_prompt(
    USER_TURN_COMPLETE_MARKER, USER_TURN_INCOMPLETE_SHORT_MARKER, USER_TURN_INCOMPLETE_LONG_MARKER
)


def _render_incomplete_long_prompt(complete: str, short: str, long: str) -> str:
    """Build the re-prompt sent when a long incomplete-turn timeout expires.

    Args:
        complete: Marker the LLM emits when the user turn is complete.
        short: Marker for a turn cut off mid-thought.
        long: Marker for a user who needs more time.

    Returns:
        The prompt text with the markers substituted in.
    """
    return f"""The user has been quiet for a while. Generate a friendly check-in message.

IMPORTANT: You MUST respond with {complete} followed by your message. Do NOT output {short} or {long} - the user has already been given plenty of time.

Your response should:
- Acknowledge they might be thinking or busy
- Offer to help or continue when ready
- Be warm and understanding
- Be brief (1 sentence)

Example format: {complete} No rush! Let me know when you're ready to continue.

Generate your {complete} response now."""


DEFAULT_INCOMPLETE_LONG_PROMPT = _render_incomplete_long_prompt(
    USER_TURN_COMPLETE_MARKER, USER_TURN_INCOMPLETE_SHORT_MARKER, USER_TURN_INCOMPLETE_LONG_MARKER
)


# System prompt instructions for turn completion that can be appended to any base prompt
def _render_completion_instructions(complete: str, short: str, long: str) -> str:
    """Build the turn completion instructions appended to the system prompt.

    Args:
        complete: Marker the LLM emits when the user turn is complete.
        short: Marker for a turn cut off mid-thought.
        long: Marker for a user who needs more time.

    Returns:
        The prompt text with the markers substituted in.
    """
    return f"""
TURN COMPLETION PROTOCOL (mandatory):
The user's words reach you from speech recognition, usually without punctuation, and sometimes before they have finished talking. Before you reply, decide whether their turn is complete, and start every response with exactly one of these markers as its very first character:

{complete}  the user's turn is complete: answer them. Write {complete}, a space, then your full reply. Never write {complete} on its own.
{short}  the user stopped mid-sentence and will continue in a few seconds. Write {short} and nothing else.
{long}  the user needs time to think or asked you to wait. Write {long} and nothing else.

Deciding:
- Complete means conversationally complete, not long. One word can be a complete answer to your question: "yes", "no thanks", "Tuesday", "Japan", "four". A question to you is complete. A correction, or a request to repeat yourself, is complete.
- Grammatically complete is not the same as conversationally complete. If the user has only acknowledged your question or reacted to what you said, without answering it ("that's a good question", "oh wow, okay", "that's interesting", "hmm", "well"), they have not taken their turn yet: {long}.
- Cut off ({short}): the last words leave a phrase open, in whatever language the user speaks: a sentence that ends on a conjunction, a preposition, an article, or the word for "because"; a list that is still going; a number that is only partly said. A fragment like this is not a request for help. Do not answer it, do not ask what they need, and do not guess the rest: wait with {short}.
- Needs time ({long}): "hold on", "let me think", "give me a second", "one moment". Filler followed by a real answer, such as "hmm, I'd go to Japan for the food", is complete.
- When the user's latest words continue an earlier fragment (your previous response was {short} or {long}), judge the fragments together as one turn. If the combined turn still ends open, {short}; if it now answers your question, {complete}.
- If a tool call is the right response, make the tool call; the turn is complete.

Format rules:
- The marker is the first character. No text, quotes, backticks or explanation before it.
- After {short} or {long}, output nothing: no words, no explanation, no nudge. The system waits and prompts you again later.
- Exactly one marker per response.

Examples:
- You asked where they would go and the user says "i'd go to japan because i love". Respond with only {short}.
- The user says "i need help with". Respond with only {short}.
- You asked for their phone number and the user says "it's five five five". Respond with only {short}.
- You asked what they want to order and the user says "a large pepperoni pizza a garden salad and". Respond with only {short}.
- You asked where they would go and the user says "that's a good question let me think". Respond with only {long}.
- You asked where they would go and the user says "that's interesting". Respond with only {long}.
- The user says "hold on a second". Respond with only {long}.
- You asked where they would go and the user says "japan". Respond with {complete} followed by your reply, for example "{complete} Japan is a wonderful choice. What draws you there?"
- You asked whether to book it and the user says "yes". Respond with {complete} followed by your reply, for example "{complete} Done, I'll book it now."
- The user says "can you help me book a flight to new york next week". Respond with {complete} followed by your reply, for example "{complete} Of course. What day would you like to leave, and from which city?"
- Your previous response was {short} after the user said "i'd go to", and now the user says "japan because". Together that is still open: respond with only {short}."""


USER_TURN_COMPLETION_INSTRUCTIONS = _render_completion_instructions(
    USER_TURN_COMPLETE_MARKER, USER_TURN_INCOMPLETE_SHORT_MARKER, USER_TURN_INCOMPLETE_LONG_MARKER
)


@dataclass
class UserTurnCompletionConfig:
    """Configuration for turn completion behavior.

    Parameters:
        instructions: Custom instructions for turn completion. If not provided,
            the default instructions are rendered from the configured markers.
        complete_marker: Marker the LLM emits when the user turn is complete.
            It is generated before any speakable text, so prefer a character
            that is a single token in the model's tokenizer.
        incomplete_short_marker: Marker for a turn cut off mid-thought.
        incomplete_long_marker: Marker for a user who needs more time.
        incomplete_short_timeout: Seconds to wait after a short incomplete
            before prompting.
        incomplete_long_timeout: Seconds to wait after a long incomplete before
            prompting.
        incomplete_short_prompt: Custom prompt when short timeout expires.
        incomplete_long_prompt: Custom prompt when long timeout expires.
    """

    instructions: str | None = None
    complete_marker: str = USER_TURN_COMPLETE_MARKER
    incomplete_short_marker: str = USER_TURN_INCOMPLETE_SHORT_MARKER
    incomplete_long_marker: str = USER_TURN_INCOMPLETE_LONG_MARKER
    incomplete_short_timeout: float = 5.0
    incomplete_long_timeout: float = 10.0
    incomplete_short_prompt: str | None = None
    incomplete_long_prompt: str | None = None

    @property
    def markers(self) -> tuple[str, str, str]:
        """The complete, short and long markers, in that order."""
        return (self.complete_marker, self.incomplete_short_marker, self.incomplete_long_marker)

    @property
    def completion_instructions(self) -> str:
        """Turn completion instructions, using default if not set."""
        return self.instructions or _render_completion_instructions(*self.markers)

    @property
    def short_prompt(self) -> str:
        """Short incomplete prompt, using default if not set."""
        return self.incomplete_short_prompt or _render_incomplete_short_prompt(*self.markers)

    @property
    def long_prompt(self) -> str:
        """Long incomplete prompt, using default if not set."""
        return self.incomplete_long_prompt or _render_incomplete_long_prompt(*self.markers)
