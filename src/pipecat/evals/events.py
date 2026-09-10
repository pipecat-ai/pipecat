#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The bot's output as the harness sees it.

:class:`EvalEventStream` turns the frames the bot-facing transport produces into
the events scenarios assert on, queues them for the matcher, and keeps the
bookkeeping the rest of the harness reads: every event seen, when each event
type last arrived (for ``send_after``), and whether the bot's next output is
still the tail of an interrupted response.

The events, and the RTVI server messages the bot emits them as:

==========================      ==============================================
scenario ``event:``             RTVI server message(s)
==========================      ==============================================
``user_started_speaking``       ``user-started-speaking``
``user_stopped_speaking``       ``user-stopped-speaking``
``vad_user_started_speaking``   ``vad-user-started-speaking`` (raw VAD, ungated by turn detection)
``vad_user_stopped_speaking``   ``vad-user-stopped-speaking`` (raw VAD, ungated by turn detection)
``user_transcription``          ``user-transcription`` (final only)
``llm_started``                 ``bot-llm-started``
``llm_response``                the LLM text: ``bot-llm-text`` joined at ``bot-llm-stopped``
``tts_response``                the TTS's spoken text: one segment per ``bot-tts-text``
                                (audio modality only)
``response``                    the harness's own transcription of the bot's audio
                                (audio modality only); ``llm_response`` in text modality
``function_call``               ``llm-function-call-in-progress``
``function_call_stopped``       ``llm-function-call-stopped``; its ``args`` carry
                                ``tool_call_id`` and ``cancelled``, so a scenario
                                can tell work that was stopped from work that
                                finished on its own
==========================      ==============================================
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

# The bot's reports that end its current turn: what it produced so far is not
# its reply to what comes next.
_INTERRUPTION_EVENTS = ("user_started_speaking", "bot_interrupted")


class EvalEventStream:
    """The bot's output as events: a queue the drivers read, fed by the pipeline.

    The sink turns each frame from the bot into an event and appends it; the
    drivers pop events as they wait for them. One rule runs through it: what
    the bot produced before the user's latest send is not the reply to it. An
    LLM response still streaming at the send, or cut by an interruption, is
    dropped until the bot's next ``llm-started``, and a spoken turn that began
    before the send is dropped when it ends.
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
        # Events are stamped in seconds since here (``at``), and a reply with the
        # arrival of its first token (``started_at``), for the timing measures.
        self._t0 = time.monotonic()
        # When either side last did anything: an event, or a frame that only
        # buffers toward one (a bot token, a persona sentence).
        self.last_activity = self._t0
        self._llm_text_at: float | None = None
        # The reply rule's flag: set by a send or an interruption, cleared at the
        # bot's next llm-started. While set, LLM text is dropped: an interrupted
        # response can still flush a trailing token after the interruption (it
        # was generated before the interrupt propagated), and that straggler
        # must not be attributed to the new turn.
        self._awaiting_reply: bool = False
        # When the driver last sent user input, and when the bot's spoken turn
        # in progress began (audio mode). A spoken turn that began before the
        # input is the bot's earlier output, not its reply, however late the
        # harness's turn analyzer finalizes it.
        self._input_sent_at: float = 0.0
        self._bot_turn_started_at: float | None = None

    async def append(self, event: dict) -> None:
        """Queue an event for the drivers.

        Stamps its arrival time as ``at`` (seconds since the stream began) unless
        it has one, records it, and logs it.

        Args:
            event: The event dict, with at least a ``type``.
        """
        event.setdefault("at", self.elapsed())
        self.touch()
        self.events_seen.append(event)
        self.latest_event_times[event["type"]] = time.monotonic()
        preview = event.get("text") or event.get("transcript") or event.get("name") or ""
        self._trace.log(f"event: {event['type']}" + (f"  {str(preview)!r}" if preview else ""))
        await self._queue.put(event)

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

        Events of other types are dropped from the queue, so a scenario need not
        list every event the bot emits; they stay in :attr:`events_seen`.

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

    def drop_pending_bot_output(self, why: str) -> None:
        """Drop the bot's queued output, so a later turn cannot match it.

        A queued ``user_transcription`` stays: a DTMF keypress reports its
        transcription just before the turn starts, and that is the turn's input,
        not stale bot output. :attr:`events_seen` and :attr:`latest_event_times`
        are left as they are.

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

    def elapsed(self) -> float:
        """Seconds since the stream began, the clock the events' ``at`` is on."""
        return round(time.monotonic() - self._t0, 3)

    def touch(self) -> None:
        """Note activity that makes no event, so a lull is measured from it."""
        self.last_activity = time.monotonic()

    def frame_to_event(self, frame: Frame) -> dict | None:
        """Translate one frame from the bot into the event it maps to, if any.

        The LLM text is buffered and emitted as one ``llm_response`` when the
        response ends; ``tts_response`` is emitted only in audio mode; the bot's
        reports about the harness (its VAD, speaking, and transcription messages)
        become their events. The audio-mode ``response`` is not made here: the
        aggregator appends it when the bot's spoken turn ends.

        Args:
            frame: A frame the bot-facing transport produced.

        Returns:
            The event the frame maps to, or ``None``; most frames map to none.
        """
        self.touch()
        if isinstance(frame, InputTransportMessageFrame):
            event = self._message_to_event(frame.message)
            if event is not None and event["type"] in _INTERRUPTION_EVENTS:
                self._interrupted()
            return event
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._awaiting_reply = False
            self._text_buffer = []
            self._llm_text_at = None
            return {"type": "llm_started"}
        elif isinstance(frame, LLMTextFrame):
            if self._awaiting_reply:
                return None
            if not self._text_buffer:
                self._llm_text_at = self.elapsed()
            self._text_buffer.append(frame.text)
            return None
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._awaiting_reply:
                self._text_buffer = []
                return None
            event = self._segment_event("llm_response", "".join(self._text_buffer))
            if self._llm_text_at is not None:
                event["started_at"] = self._llm_text_at
            return event
        elif isinstance(frame, TTSTextFrame):
            if self._bot_audio:
                return self._segment_event("tts_response", frame.text)
            return None
        elif isinstance(frame, FunctionCallInProgressFrame):
            return {
                "type": "function_call",
                "name": frame.function_name or None,
                "args": dict(frame.arguments or {}),
            }
        elif isinstance(frame, (FunctionCallResultFrame, FunctionCallCancelFrame)):
            # How the call ended is the assertable part, so `cancelled` sits in
            # `args` alongside the id: a scenario matches both through the same
            # `calls:`/`args:` check a function_call uses.
            return {
                "type": "function_call_stopped",
                "name": frame.function_name or None,
                "args": {
                    "tool_call_id": frame.tool_call_id,
                    "cancelled": isinstance(frame, FunctionCallCancelFrame),
                },
            }
        return None

    def bot_turn_started(self) -> None:
        """Note that the bot began a spoken turn."""
        self._bot_turn_started_at = time.monotonic()

    async def bot_turn_stopped(self, text: str) -> None:
        """Append the bot's finished spoken turn as a ``response``, unless it is stale.

        A turn is stale when it began before the user's latest send, or when the
        bot's LLM has not restarted since. An interrupted turn can finalize after
        the bot has begun its real reply, and must not pass for it.

        Args:
            text: The turn's transcription; nothing is appended when empty.
        """
        started = self._bot_turn_started_at
        self._bot_turn_started_at = None
        if not text:
            return
        if self._awaiting_reply or (started is not None and started < self._input_sent_at):
            self._trace.log(f"discard: bot turn from before the send {text!r}")
            return
        await self.append({"type": "response", "text": text})

    def input_sent(self) -> None:
        """Mark the user's input as sent: the bot's reply is what it says from now on."""
        self._awaiting_reply = True
        self._input_sent_at = time.monotonic()

    def _interrupted(self) -> None:
        """The bot reported an interruption: drop its pending output and wait for a fresh reply."""
        self.drop_pending_bot_output("on interruption")
        self._awaiting_reply = True

    def _message_to_event(self, message) -> dict | None:
        """Map one of the bot's reports about the harness to its event, if any."""
        if not isinstance(message, dict):
            return None
        msg_type = message.get("type")
        data = message.get("data") or {}
        if msg_type == "user-started-speaking":
            return {"type": "user_started_speaking"}
        elif msg_type == "bot-interrupted":
            return {"type": "bot_interrupted"}
        elif msg_type == "user-stopped-speaking":
            return {"type": "user_stopped_speaking"}
        elif msg_type == "vad-user-started-speaking":
            return {"type": "vad_user_started_speaking"}
        elif msg_type == "vad-user-stopped-speaking":
            return {"type": "vad_user_stopped_speaking"}
        elif msg_type == "user-transcription":
            if data.get("final", True):
                return {"type": "user_transcription", "transcript": data.get("text", "")}
            return None
        return None

    def _segment_event(self, event_type: str, text: str) -> dict:
        """One response segment; the text may be empty, as after an interruption."""
        return {"type": event_type, "text": text}
