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
``user_transcription``          ``user-input`` of type ``transcription`` (final only)
``llm_started``                 ``bot-llm-started``
``llm_response``                the LLM text: ``bot-llm-text`` joined at ``bot-llm-stopped``
``llm_marker``                  ``bot-llm-marker``: the sideband marker the LLM emitted in a
                                response, such as a turn-completion marker, with its
                                ``kind`` and the response's raw text; sent when the
                                response ends, only on request
``tts_response``                the TTS's spoken text: one segment per ``bot-tts-text``
                                (audio modality only)
``response``                    the harness's own transcription of the bot's audio
                                (audio modality only); ``llm_response`` in text modality
``function_call``               ``llm-function-call-in-progress``; a call made
                                outside the bot's pipeline and reported into it
                                (``ExternalFunctionCall*Frame``), such as a
                                backend's, counts the same
``function_call_stopped``       ``llm-function-call-stopped``; its ``args`` carry
                                ``tool_call_id`` and ``cancelled``, so a scenario
                                can tell work that was stopped from work that
                                finished on its own
``image``                       ``eval-bot-image``: an image the bot output, with
                                its ``width``, ``height``, and ``format``; sent
                                only on request
``bot_backchannel``             ``bot-output`` of type ``backchannel``, once spoken
                                to the end: a short acknowledgment the bot said
                                while the user talked, such as "Mm-hmm."
``user_backchannel``            ``user-input`` of type ``backchannel``: a short
                                acknowledgment the user said while the bot
                                talked, which neither interrupted it nor reached
                                the LLM
==========================      ==============================================
"""

import asyncio
import math
import time
from collections.abc import Callable

from pipecat.evals.results import EvalTrace
from pipecat.evals.serializer import EVAL_BOT_IMAGE_TYPE
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ExternalFunctionCallCancelFrame,
    ExternalFunctionCallInProgressFrame,
    ExternalFunctionCallResultFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputTransportMessageFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMarkerResponseFrame,
    LLMTextFrame,
    TTSTextFrame,
)

# The bot's reports that end its current turn: what it produced so far is not
# its reply to what comes next.
_INTERRUPTION_EVENTS = ("user_started_speaking", "bot_interrupted")


class EvalBotEndedError(Exception):
    """The bot ended the call before the awaited event arrived, so it never will."""


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
        # The events popped during the current turn, in order, and which of them
        # an expectation has claimed. An expectation looks for its event from
        # the point the turn had reached when it began, or from the start of the
        # turn (``check_from: turn_start``), so an event popped on the way to
        # another expectation's is not lost to one that may look back for it.
        self._turn_log: list[dict] = []
        self._claimed: set[int] = set()
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
        # When the driver last sent user input, and, when that send talked over
        # the bot, when the bot started speaking again (infinity until it
        # does): its speech from before then is what it was saying, not its
        # reply, however late its transcription lands (audio mode).
        self._input_sent_at: float = 0.0
        self._talked_over_until: float | None = None
        # Whether the turn sends nothing and waits for the bot's next speech,
        # which is its reply whether or not the bot's LLM produced it (a
        # presence check, say). Cleared when the bot starts speaking.
        self._observing: bool = False
        # Whether the bot is speaking, by its own report: cleared when it
        # starts, set when it stops or is interrupted. An event, so a driver
        # that must not talk over the bot can wait for it without polling.
        self._bot_quiet = asyncio.Event()
        self._bot_quiet.set()
        # Whether everything the bot has said is transcribed and closed into a
        # turn (audio mode): a segment of its speech the VAD still hears,
        # segments still to be transcribed, and whether a spoken turn is open.
        # An event, so a driver can wait for the bot's speech to be fully
        # accounted for before it sends.
        self._bot_segment_open = False
        self._bot_segments_pending = 0
        self._bot_turn_open = False
        self._bot_transcribed = asyncio.Event()
        self._bot_transcribed.set()
        # Whether the bot closed the connection. Kept apart from its queued
        # bot_ended event, which a drop of the bot's pending output discards.
        self._bot_ended = False

    @property
    def bot_speaking(self) -> bool:
        """Whether the bot reports itself speaking right now."""
        return not self._bot_quiet.is_set()

    @property
    def talked_over_until(self) -> float | None:
        """Monotonic time the bot started speaking again after the latest turn began.

        The bot's speech that began before then belongs to the turn before:
        what it was saying when a send talked over it, or the reply an observing
        turn waited out. Infinity until the bot starts speaking again, and
        ``None`` when the latest send did not talk over it.
        """
        return self._talked_over_until

    @property
    def bot_transcribed(self) -> bool:
        """Whether everything the bot has said is transcribed and closed into a turn."""
        return self._bot_transcribed.is_set()

    @property
    def bot_has_ended(self) -> bool:
        """Whether the bot ended the call (see :meth:`bot_ended`)."""
        return self._bot_ended

    @property
    def bot_speech_outstanding(self) -> str:
        """What of the bot's speech is not yet accounted for, for the trace."""
        parts = []
        if self._bot_segment_open:
            parts.append("a segment still being heard")
        if self._bot_segments_pending:
            parts.append(f"{self._bot_segments_pending} segment(s) awaiting transcription")
        if self._bot_turn_open:
            parts.append("a turn still open")
        return ", ".join(parts) or "nothing"

    async def wait_bot_transcribed(self, timeout: float) -> bool:
        """Wait for everything the bot has said to be transcribed and closed into a turn.

        Args:
            timeout: Seconds to wait at most.

        Returns:
            Whether the bot's speech is fully accounted for, or False if some
            of it was still being transcribed at the timeout.
        """
        try:
            await asyncio.wait_for(self._bot_transcribed.wait(), timeout)
        except TimeoutError:
            return False
        return True

    def bot_segment_started(self) -> None:
        """The VAD hears a segment of the bot's speech."""
        self._bot_segment_open = True
        self._bot_transcribed.clear()

    def bot_segment_ended(self) -> None:
        """A segment of the bot's speech ended and awaits transcription."""
        self._bot_segment_open = False
        self._bot_segments_pending += 1
        self._bot_transcribed.clear()

    def bot_segment_transcribed(self) -> None:
        """A segment of the bot's speech was transcribed, whether or not it yielded text."""
        self._bot_segments_pending = max(0, self._bot_segments_pending - 1)
        self._update_bot_transcribed()

    def bot_turn_started(self) -> None:
        """The bot began a spoken turn."""
        self._bot_turn_open = True
        self._bot_transcribed.clear()

    def bot_ended(self) -> None:
        """The bot closed the connection: it sends nothing more, and is no longer speaking."""
        self._bot_ended = True
        self._bot_quiet.set()

    def _update_bot_transcribed(self) -> None:
        if (
            not self._bot_segment_open
            and self._bot_segments_pending == 0
            and not self._bot_turn_open
        ):
            self._bot_transcribed.set()

    async def wait_bot_quiet(self, timeout: float) -> bool:
        """Wait for the bot to stop speaking.

        Args:
            timeout: Seconds to wait at most.

        Returns:
            Whether the bot is quiet, or False if it was still speaking at the
            timeout.
        """
        try:
            await asyncio.wait_for(self._bot_quiet.wait(), timeout)
        except TimeoutError:
            return False
        return True

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

    @property
    def turn_position(self) -> int:
        """How many events the turn has popped so far: where an expectation beginning now looks from."""
        return len(self._turn_log)

    def begin_turn(self) -> None:
        """Forget the events popped during the previous turn."""
        self._turn_log = []
        self._claimed = set()

    def unclaimed_since(self, since: int) -> list[dict]:
        """The events popped from position ``since`` on that no expectation has claimed."""
        return [
            event for i, event in enumerate(self._turn_log) if i >= since and i not in self._claimed
        ]

    async def next_event(
        self,
        event_type: str,
        deadline: float,
        *,
        since: int | None = None,
        accept: Callable[[dict], bool] | None = None,
    ) -> dict:
        """The next unclaimed event of ``event_type`` that ``accept`` takes, looking from ``since``.

        The turn's log is searched first, from ``since`` on, then the queue is
        popped until such an event arrives. Events popped on the way stay in the
        log, unclaimed: a scenario need not list every event the bot emits, and
        an expectation that looks back from an earlier point can still claim
        them. Every event stays in :attr:`events_seen`.

        Args:
            event_type: The event type to wait for.
            deadline: Monotonic time to give up at.
            since: Position in the turn's log to look from; ``None`` looks only
                at what arrives from now on.
            accept: A further check on an event of the type, such as a call's
                name; one it turns down is left for another expectation.

        Returns:
            The event.

        Raises:
            TimeoutError: If none arrives before ``deadline``.
            EvalBotEndedError: If the bot ended the call and none is left to
                arrive. The harness's transcription of what the bot said last
                (audio mode) is waited for first, since it can land after the
                bot is gone.
        """

        def wanted(event: dict) -> bool:
            return event.get("type") == event_type and (accept is None or accept(event))

        if since is not None:
            for i in range(since, len(self._turn_log)):
                if i not in self._claimed and wanted(self._turn_log[i]):
                    self._claimed.add(i)
                    self._trace.log(f"match: {event_type!r} from earlier in the turn")
                    return self._turn_log[i]
        while True:
            if self._bot_ended and self._queue.empty():
                if not await self.wait_bot_transcribed(deadline - time.monotonic()):
                    raise TimeoutError()
                # A transcription that yields text queues its event as it closes.
                if self._queue.empty():
                    raise EvalBotEndedError()
            event = await self.next_any(deadline)
            self._turn_log.append(event)
            if wanted(event):
                self._claimed.add(len(self._turn_log) - 1)
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
        response ends; a marker is its own ``llm_marker`` as soon as it
        arrives; ``tts_response`` is emitted only in audio mode; the bot's
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
        elif isinstance(frame, LLMMarkerResponseFrame):
            if self._awaiting_reply:
                return None
            return {
                "type": "llm_marker",
                "text": frame.marker or "",
                "kind": frame.kind,
                "raw": frame.raw,
                "markers": list(frame.markers),
            }
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
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_quiet.clear()
            if self._talked_over_until == math.inf:
                self._talked_over_until = time.monotonic()
            if self._observing:
                self._observing = False
                self._awaiting_reply = False
            return {"type": "bot_started_speaking"}
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_quiet.set()
            return {"type": "bot_stopped_speaking"}
        elif isinstance(frame, (FunctionCallInProgressFrame, ExternalFunctionCallInProgressFrame)):
            return {
                "type": "function_call",
                "name": frame.function_name or None,
                "args": dict(frame.arguments or {}),
            }
        elif isinstance(frame, ExternalFunctionCallResultFrame) and not frame.is_final:
            return None
        elif isinstance(
            frame,
            (
                FunctionCallResultFrame,
                FunctionCallCancelFrame,
                ExternalFunctionCallResultFrame,
                ExternalFunctionCallCancelFrame,
            ),
        ):
            # How the call ended is the assertable part, so `cancelled` sits in
            # `args` alongside the id: a scenario matches both through the same
            # `calls:`/`args:` check a function_call uses.
            return {
                "type": "function_call_stopped",
                "name": frame.function_name or None,
                "args": {
                    "tool_call_id": frame.tool_call_id,
                    "cancelled": isinstance(
                        frame, (FunctionCallCancelFrame, ExternalFunctionCallCancelFrame)
                    ),
                },
            }
        return None

    @property
    def input_sent_at(self) -> float:
        """Monotonic time of the user's latest send; bot speech from before it is not the reply."""
        return self._input_sent_at

    async def bot_turn_stopped(self, text: str) -> None:
        """Append the bot's finished spoken turn as a ``response``, unless it is stale.

        Speech the bot was saying when a send talked over it never reaches a
        turn (the client drops it by its audio's time). A turn is still stale while the
        bot's LLM has not restarted since the send: the bot going on with what
        it was saying, which must not pass for the reply.

        Args:
            text: The turn's transcription; nothing is appended when empty.
        """
        self._bot_turn_open = False
        self._update_bot_transcribed()
        if not text:
            return
        if self._awaiting_reply:
            self._trace.log(f"discard: bot turn from before the reply {text!r}")
            return
        await self.append({"type": "response", "text": text})

    def input_sent(self) -> None:
        """Mark the user's input as sent: the bot's reply is what it says from now on."""
        self.turn_boundary()

    def turn_boundary(self) -> None:
        """Start a new turn: only what the bot says from now on belongs to it.

        When the bot is speaking at this point, the turn talks over it, and what
        the bot says until it starts speaking again, however late its
        transcription lands, is the previous turn's and is discarded.
        """
        self._awaiting_reply = True
        self._input_sent_at = time.monotonic()
        self._talked_over_until = math.inf if self.bot_speaking else None
        self._observing = False

    def observing_boundary(self) -> None:
        """Start a turn that sends nothing: what the bot says next belongs to it.

        What the bot says until it starts speaking again, however late its
        transcription lands, is the previous turn's and is discarded. What it
        says from then on is the turn's, even when its LLM did not produce it.
        """
        self.turn_boundary()
        self._talked_over_until = math.inf
        self._observing = True

    def _interrupted(self) -> None:
        """The bot reported an interruption: drop its pending output and wait for a fresh reply."""
        self.drop_pending_bot_output("on interruption")
        self._awaiting_reply = True
        self._bot_quiet.set()

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
        elif msg_type == "bot-output":
            return self._bot_backchannel_event(data)
        elif msg_type == "user-input":
            return self._user_input_event(data)
        elif msg_type == EVAL_BOT_IMAGE_TYPE:
            return {
                "type": "image",
                "width": data.get("width"),
                "height": data.get("height"),
                "format": data.get("format"),
            }
        return None

    def _user_input_event(self, data: dict) -> dict | None:
        """The user's final transcription or backchannel, from a ``user-input``."""
        input_type = data.get("input_type")
        if input_type == "transcription" and data.get("final"):
            return {"type": "user_transcription", "transcript": data.get("text", "")}
        elif input_type == "backchannel":
            return {"type": "user_backchannel", "text": data.get("text", "")}
        return None

    def _bot_backchannel_event(self, data: dict) -> dict | None:
        """The bot's backchannel, from the ``bot-output`` reporting it was spoken to the end."""
        if data.get("output_type") != "backchannel" or data.get("spoken_status") != "completed":
            return None
        progress = data.get("spoken_progress") or {}
        return {
            "type": "bot_backchannel",
            "text": progress.get("accumulated_text") or data.get("text", ""),
        }

    def _segment_event(self, event_type: str, text: str) -> dict:
        """One response segment; the text may be empty, as after an interruption."""
        return {"type": event_type, "text": text}
