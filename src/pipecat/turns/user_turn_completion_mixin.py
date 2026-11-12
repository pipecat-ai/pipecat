#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mixin for adding turn completion detection to LLM services.

This mixin lets LLM services read the turn completion markers (●, ◐, ○) an LLM
starts each response with, and report them so a user turn stop strategy can
decide whether the user was done.
"""

from dataclasses import dataclass
from enum import Enum

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMarkerFrame,
    LLMMarkerResponseFrame,
    LLMTextFrame,
    UserTurnCompletionFrame,
    UserTurnInferenceCompletedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

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


class UserTurnCompletionLLMServiceMixin(FrameProcessor):
    """Mixin that reads turn completion markers in LLM responses.

    Each response starts with a marker: ● (the turn is complete), ◐ (the user
    will continue shortly) or ○ (the user needs more time). The mixin strips
    the marker from the spoken text and reports it:

    - upstream, once per inference, as a ``UserTurnCompletionFrame`` for the
      ``LLMTurnCompletionUserTurnStopStrategy`` to decide on. A tool call
      before any marker reports a complete turn, and a response without a
      marker reports no kind (``None``);
    - in both directions as a ``UserTurnInferenceCompletedFrame`` when the
      turn is complete;
    - downstream as an ``LLMMarkerFrame``, which the assistant aggregator
      writes to the context.

    Text after ◐ or ○ is not spoken: the marker is the whole response.

    Usage example::

        # With turn completion:
        if self._filter_incomplete_user_turns:
            await self._push_turn_text(chunk.text)
        else:
            await self.push_frame(LLMTextFrame(chunk.text))

    The LLM service controls when to use turn completion by calling
    ``_push_turn_text`` instead of ``push_frame``.

    The host calls ``_track_pushed_frame`` with every frame it pushes, before
    anything decides whether the frame goes out, as ``LLMService.push_frame``
    does.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the turn completion mixin.

        Args:
            *args: Positional arguments passed to parent class.
            **kwargs: Keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        # Enabled by LLMTurnCompletionUserTurnStopStrategy over an LLMUpdateSettingsFrame
        # once the pipeline starts.
        self._filter_incomplete_user_turns: bool = False
        # The LLMContextFrame that started the current inference, and whether
        # its turn-completion judgment has been reported upstream yet.
        self._context_frame_id: int | None = None
        self._turn_completion_reported = False
        self._turn_text_buffer = ""
        # The current response as the LLM produced it, and the marker read
        # from it, reported together when the response ends.
        self._response_raw = ""
        self._response_marker: tuple[str, str] | None = None
        # Completion verdict for the current LLM response, set when a marker is
        # detected in the text stream. ``None`` means no marker yet, so keep
        # buffering text until one appears. ``INCOMPLETE`` also doubles as a
        # safety latch: the prompt tells the LLM to emit only the marker, but if
        # it disobeys and streams more text we keep suppressing it.
        self._turn_marker: TurnMarker | None = None
        self._user_turn_completion_config = UserTurnCompletionConfig()

    def set_user_turn_completion_config(self, config: UserTurnCompletionConfig | None):
        """Set the turn completion configuration.

        Args:
            config: The turn completion configuration, or None to restore the
                default configuration.
        """
        self._user_turn_completion_config = config or UserTurnCompletionConfig()

    async def _turn_reset(self):
        """Reset turn completion state between responses.

        Call this at the end of each LLM response to clear buffered text and reset state.
        If no marker was found, pushes the buffered text to avoid losing content.
        """
        # If no marker was found in this response, push the buffered text so
        # it's not lost.
        if self._turn_marker is None and self._turn_text_buffer:
            # Graceful degradation: push the buffered text so it's not lost
            logger.warning(
                f"{self}: filter_incomplete_user_turns is enabled but LLM response did not "
                f"contain turn completion markers (●/◐/○). Pushing text anyway. "
                "The system prompt may be missing turn completion instructions."
            )
            await self.push_frame(LLMTextFrame(self._turn_text_buffer))

        self._turn_text_buffer = ""
        self._turn_marker = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, handling turn completion state resets.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        if isinstance(frame, InterruptionFrame):
            await self._turn_reset()
        elif isinstance(frame, LLMContextFrame):
            self._context_frame_id = frame.id
            self._turn_completion_reported = False

        # Pass frame to parent
        await super().process_frame(frame, direction)

    async def _track_pushed_frame(self, frame: Frame):
        """Follow the response the service is pushing, resetting at its end.

        The service calls this for every frame it pushes, before anything
        decides whether the frame goes out: a speculative response the gate
        drops still ends here, and its marker is reported while the turn
        waits for it.

        Args:
            frame: The frame the service is about to push.
        """
        if isinstance(frame, FunctionCallsStartedFrame):
            # A tool call is the LLM judging the turn complete. Reported before
            # the function dispatches, so the turn can end before its result
            # reaches the assistant aggregator.
            if self._filter_incomplete_user_turns:
                await self._report_turn_completion(TurnMarker.COMPLETE.value, None)
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._response_raw = ""
            self._response_marker = None
        elif isinstance(frame, LLMFullResponseEndFrame):
            # A response that ends without a marker or a tool call carries no judgment.
            if self._filter_incomplete_user_turns:
                await self._report_turn_completion(None, None)
            await self._report_response()
            await self._turn_reset()

    async def _push_marker(self, marker: str, kind: str):
        """Report the marker read from the response, and push it for the context."""
        self._response_marker = (marker, kind)
        await self._report_turn_completion(kind, marker)
        # Only a complete turn's marker belongs to the response that follows.
        await self.push_frame(
            LLMMarkerFrame(marker, append_to_context_immediately=kind != "complete")
        )

    async def _report_turn_completion(self, kind: str | None, marker: str | None):
        """Push the turn-completion judgment upstream, once per inference.

        A complete turn is also broadcast as a ``UserTurnInferenceCompletedFrame``.
        """
        if self._turn_completion_reported:
            return
        self._turn_completion_reported = True
        await self.push_frame(
            UserTurnCompletionFrame(
                kind=kind, marker=marker, context_frame_id=self._context_frame_id
            ),
            FrameDirection.UPSTREAM,
        )
        if kind == TurnMarker.COMPLETE.value:
            await self.broadcast_frame(UserTurnInferenceCompletedFrame)

    async def _report_response(self):
        """Push the response's raw text and marker, once there was a response to read."""
        if not self._response_raw and self._response_marker is None:
            return
        config = self._user_turn_completion_config
        marker, kind = self._response_marker or (None, None)
        await self.push_frame(
            LLMMarkerResponseFrame(
                raw=self._response_raw,
                marker=marker,
                kind=kind,
                markers=[
                    config.complete_marker,
                    config.incomplete_short_marker,
                    config.incomplete_long_marker,
                ],
            )
        )
        self._response_raw = ""
        self._response_marker = None

    async def _push_turn_text(self, text: str):
        """Push LLM text with turn completion detection.

        This method should be used instead of `push_frame(LLMTextFrame(text))` when
        turn completion is enabled. It will:
        1. Buffer text until a turn marker (●, ◐, or ○) appears
        2. When ◐ or ○ is found: report it, and drop the rest of the response
        3. When ● is found: report it, and push the text after it
        4. After ●: all subsequent text flows through immediately

        Args:
            text: The text content from the LLM to push.
        """
        self._response_raw += text

        # If we've already detected incomplete, suppress all remaining text.
        # This is a safety mechanism in case the LLM disobeys the prompt and outputs
        # additional text after the marker (e.g., "◐ Please continue...").
        if self._turn_marker == TurnMarker.INCOMPLETE:
            return

        # If ● (COMPLETE) was already found, push text immediately without buffering
        if self._turn_marker == TurnMarker.COMPLETE:
            await self.push_frame(LLMTextFrame(text))
            return

        config = self._user_turn_completion_config

        # Add text to buffer
        self._turn_text_buffer += text

        # Check for incomplete markers (◐ short, ○ long): the user was cut off or
        # needs time, and the marker is the whole response.
        incomplete_type: IncompleteType | None = None
        if config.incomplete_short_marker in self._turn_text_buffer:
            incomplete_type = IncompleteType.SHORT
        elif config.incomplete_long_marker in self._turn_text_buffer:
            incomplete_type = IncompleteType.LONG

        if incomplete_type:
            marker = (
                config.incomplete_short_marker
                if incomplete_type == IncompleteType.SHORT
                else config.incomplete_long_marker
            )
            logger.debug(
                f"INCOMPLETE {incomplete_type.value.upper()} ({marker}) detected, suppressing text"
            )
            self._turn_marker = TurnMarker.INCOMPLETE

            # Persist the marker to context as a stand-alone assistant
            # message via LLMMarkerFrame: the bot produces no spoken
            # output for incomplete turns, so the marker is the entire
            # context entry.
            await self._push_marker(marker, incomplete_type.value)

            self._turn_text_buffer = ""
            return

        # Check for ● (COMPLETE) marker - user's turn was complete, respond normally
        if config.complete_marker in self._turn_text_buffer:
            logger.debug(f"COMPLETE ({config.complete_marker}) detected, pushing buffered text")

            # Push the marker as a sideband signal that the assistant
            # aggregator will prepend to the upcoming aggregated text,
            # so the context message ends up as "● <response>".
            await self._push_marker(config.complete_marker, TurnMarker.COMPLETE.value)

            # Split buffer at the marker to handle cases where marker and text
            # arrive in the same chunk (e.g., "● Hello!" from some LLMs)
            marker_pos = self._turn_text_buffer.index(config.complete_marker)
            marker_end = marker_pos + len(config.complete_marker)

            # Push remaining text after marker as normal speech
            remaining_text = self._turn_text_buffer[marker_end:]
            if remaining_text:
                # Strip leading space after marker if present (● Hello -> Hello)
                if remaining_text.startswith(" "):
                    remaining_text = remaining_text[1:]
                if remaining_text:
                    await self.push_frame(LLMTextFrame(remaining_text))

            # Mark complete - all subsequent text flows through immediately
            self._turn_text_buffer = ""
            self._turn_marker = TurnMarker.COMPLETE
            return
