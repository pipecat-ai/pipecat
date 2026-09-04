#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The bot's output as the harness sees it.

:class:`EvalEventStream` turns the frames the bot-facing transport produces into
the friendly events scenarios assert on, queues them for the matcher, and keeps
the bookkeeping the rest of the harness reads: every event seen, when each event
type last arrived (for ``send_after``), and whether the bot's next output is
still the tail of an interrupted response.
"""

import asyncio
import time

from pipecat.evals.results import EvalTrace
from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputTransportMessageFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSTextFrame,
)


class EvalEventStream:
    """Translates the bot's frames into scenario events and queues them.

    The pipeline's bot-frame sink feeds every frame the transport produces
    through :meth:`frames_to_events` and appends the results with
    :meth:`append`; the matcher consumes them with :meth:`next_event` and
    :meth:`next_any`. The audio-mode ``response`` is appended by the user
    aggregator when the bot's spoken turn ends.
    """

    def __init__(self, *, bot_audio: bool, trace: EvalTrace):
        """Initialize the stream.

        Args:
            bot_audio: Whether the scenario runs in audio mode; ``tts_response``
                is emitted only then.
            trace: The run's trace, where each event is logged as it arrives.
        """
        self._bot_audio = bot_audio
        self._trace = trace
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        # Every event in arrival order, for the result's diagnostics.
        self.events_seen: list[dict] = []
        # When each event type last arrived, for send_after anchoring.
        self.latest_event_times: dict[str, float] = {}
        # Accumulates the bot's LLM text for the current response, emitted as one
        # llm_response segment when the response ends.
        self._text_buffer: list[str] = []
        # Set on an interruption (and by the driver after a send), cleared at the
        # bot's next llm-started. While set, llm_response segments are dropped:
        # the interrupted response can still flush a trailing token *after* the
        # interruption event (it was generated before the interrupt propagated),
        # and that straggler must not be attributed to the new turn. The genuinely
        # new response begins at the next llm-started.
        self.awaiting_llm_restart: bool = False

    async def append(self, event: dict) -> None:
        """Append an event for the matcher.

        Records it in :attr:`events_seen`, stamps its arrival time in
        :attr:`latest_event_times`, and queues it.

        Args:
            event: The event dict, with at least a ``type``.
        """
        self.events_seen.append(event)
        self.latest_event_times[event["type"]] = time.monotonic()
        preview = event.get("text") or event.get("transcript") or event.get("name") or ""
        self._trace.log(f"event: {event['type']}" + (f"  {str(preview)!r}" if preview else ""))
        await self._queue.put(event)

    def frames_to_events(self, frame: Frame) -> list[dict]:
        """Translate one incoming pipeline frame into zero or more friendly events.

        The :class:`~pipecat.transports.websocket.rtvi_client.RTVIClientTransport`
        deserializes the bot's RTVI server messages into frames; this maps those
        frames to the events the matcher consumes, applying the modality/aggregation
        rules (buffer the LLM text, suppress an interrupted response's straggler,
        emit ``tts_response`` only in audio mode, etc.).

        The bot *reports* events about the harness as ``InputTransportMessageFrame``
        (see :data:`~pipecat.evals.serializer.RTVIHarnessSerializer`), which this
        maps to scenario events. What the harness *computes* from the bot's audio
        is handled elsewhere: the ``response`` comes from the user aggregator's
        ``on_user_turn_stopped`` (it consumes the STT's ``TranscriptionFrame``s, so
        they never reach here), and the aggregator's own VAD/speaking frames are
        internal plumbing, ignored here.

        Args:
            frame: A frame the bot-facing transport produced.

        Returns:
            The events the frame maps to, in order (often none).
        """
        if isinstance(frame, InputTransportMessageFrame):
            return self._message_to_events(frame.message)
        if isinstance(frame, LLMFullResponseStartFrame):
            self.awaiting_llm_restart = False
            self._text_buffer = []
            return [{"type": "llm_started"}]
        if isinstance(frame, LLMTextFrame):
            if self.awaiting_llm_restart:
                return []
            self._text_buffer.append(frame.text)
            return []
        if isinstance(frame, LLMFullResponseEndFrame):
            if self.awaiting_llm_restart:
                self._text_buffer = []
                return []
            return [self._segment_event("llm_response", "".join(self._text_buffer))]
        if isinstance(frame, TTSTextFrame):
            if self._bot_audio:
                return [self._segment_event("tts_response", frame.text)]
            return []
        if isinstance(frame, FunctionCallInProgressFrame):
            return [
                {
                    "type": "function_call",
                    "name": frame.function_name or None,
                    "args": dict(frame.arguments or {}),
                }
            ]
        if isinstance(frame, (FunctionCallResultFrame, FunctionCallCancelFrame)):
            # How the call ended is the assertable part, so `cancelled` sits in
            # `args` alongside the id: a scenario matches both through the same
            # `calls:`/`args:` check a function_call uses.
            return [
                {
                    "type": "function_call_stopped",
                    "name": frame.function_name or None,
                    "args": {
                        "tool_call_id": frame.tool_call_id,
                        "cancelled": isinstance(frame, FunctionCallCancelFrame),
                    },
                }
            ]
        return []

    def drop_pending_bot_output(self, why: str) -> None:
        """Drop the bot's un-matched output, so a later turn can't match it.

        Clears the response buffer and drains the bot's pending output from the
        queue, so a greeting (or any prior bot output) can't be matched against
        the next turn. ``user_transcription`` is preserved: a DTMF keypress emits
        its transcription immediately before the turn-start interruption, and
        that transcription is the turn's *input*, not the stale bot output this
        drop is meant to clear. Diagnostics (:attr:`events_seen`,
        :attr:`latest_event_times`) are left intact for ``send_after`` lookups.

        Args:
            why: What prompted the drop, for the trace.
        """
        self._text_buffer = []
        preserved: list[dict] = []
        dropped = 0
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if event.get("type") == "user_transcription":
                preserved.append(event)
            else:
                dropped += 1
        for event in preserved:
            self._queue.put_nowait(event)
        if dropped:
            self._trace.log(f"discard: dropped {dropped} queued event(s) {why}")

    async def next_any(self, deadline: float) -> dict:
        """Pop the next queued event, whatever its type.

        Args:
            deadline: Monotonic time to give up at.

        Returns:
            The event.

        Raises:
            TimeoutError: If no event arrives before ``deadline``.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError()
        async with asyncio.timeout(remaining):
            return await self._queue.get()

    async def next_event(self, event_type: str, deadline: float) -> dict:
        """Pop events until one of ``event_type`` arrives.

        Events of other types are dropped, so a scenario doesn't have to
        enumerate every event the bot emits. They remain in :attr:`events_seen`
        and :attr:`latest_event_times` for diagnostics and ``send_after`` lookups.

        Args:
            event_type: The event type to wait for.
            deadline: Monotonic time to give up at.

        Returns:
            The first event of that type.

        Raises:
            TimeoutError: If none arrives before ``deadline``.
        """
        while True:
            event = await self.next_any(deadline)
            if event.get("type") == event_type:
                return event

    def _message_to_events(self, message) -> list[dict]:
        """Map one of the bot's reported RTVI messages to scenario events.

        These are the bot's reports *about the harness* (its raw VAD, turn-level
        speaking, and the transcription of what it heard), kept as raw messages by
        the harness serializer so they don't collide with the VAD/transcription
        frames the harness computes from the bot's audio.
        """
        if not isinstance(message, dict):
            return []
        msg_type = message.get("type")
        data = message.get("data") or {}
        if msg_type == "user-started-speaking":
            # A new user turn: drop any leftover bot output from a prior turn so it
            # isn't aggregated into this one.
            self.drop_pending_bot_output("on interruption")
            self.awaiting_llm_restart = True
            return [{"type": "user_started_speaking"}]
        if msg_type == "bot-interrupted":
            self.drop_pending_bot_output("on interruption")
            self.awaiting_llm_restart = True
            return [{"type": "bot_interrupted"}]
        if msg_type == "user-stopped-speaking":
            return [{"type": "user_stopped_speaking"}]
        if msg_type == "vad-user-started-speaking":
            return [{"type": "vad_user_started_speaking"}]
        if msg_type == "vad-user-stopped-speaking":
            return [{"type": "vad_user_stopped_speaking"}]
        if msg_type == "user-transcription":
            if data.get("final", True):
                return [{"type": "user_transcription", "transcript": data.get("text", "")}]
            return []
        return []

    def _segment_event(self, event_type: str, text: str) -> dict:
        """Build one response segment of ``event_type``.

        Used for ``llm_response`` (the LLM text) and ``tts_response`` (the TTS's
        spoken text). The text may be empty (e.g. an interrupted response); the
        matcher aggregates successive segments until the content check passes.
        """
        return {"type": event_type, "text": text}
