#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram Voice Agent API protocol for a Pipecat pipeline.

:class:`DeepgramVoiceAgentProcessor` handles the client's control messages and
sends the server's messages; :class:`DeepgramVoiceAgentObserver` watches the
pipeline and reports what happens in it as Voice Agent events. Together with
:class:`~pipecat.extensions.deepgram_voice_agent.serializer.DeepgramVoiceAgentSerializer`
they let a client written for the Voice Agent API talk to a Pipecat pipeline.
"""

import asyncio
import json
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

from pipecat.extensions.deepgram_voice_agent.services import SettingsError, speak_settings
from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    ClientFunctionCallRequestFrame,
    ClientFunctionCallResultFrame,
    EndFrame,
    EndWorkerFrame,
    ErrorFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import LLMSettings
from pipecat.transports.base_output import BaseOutputTransport

if TYPE_CHECKING:
    from pipecat.extensions.deepgram_voice_agent.listen import DeepgramListen, DeepgramListenUpdater

# Error codes the Voice Agent API reports.
NON_SETTINGS_MESSAGE_BEFORE_SETTINGS = "NON_SETTINGS_MESSAGE_BEFORE_SETTINGS"
SETTINGS_ALREADY_APPLIED = "SETTINGS_ALREADY_APPLIED"
UNPARSABLE_CLIENT_MESSAGE = "UNPARSABLE_CLIENT_MESSAGE"
INVALID_SETTINGS = "INVALID_SETTINGS"
UNSUPPORTED_UPDATE = "UNSUPPORTED_UPDATE"
PIPELINE_ERROR = "PIPELINE_ERROR"
SPEAK_REQUEST_FAILED = "SPEAK_REQUEST_FAILED"
FORCE_END_TURN_UNSUPPORTED = "FORCE_END_TURN_UNSUPPORTED"
CLIENT_MESSAGE_TIMEOUT = "CLIENT_MESSAGE_TIMEOUT"
INJECT_AGENT_MESSAGE_DURING_USER_SPEECH = "INJECT_AGENT_MESSAGE_DURING_USER_SPEECH"
INJECT_AGENT_MESSAGE_DURING_AGENT_SPEECH = "INJECT_AGENT_MESSAGE_DURING_AGENT_SPEECH"

CLIENT_MESSAGE_TYPES = {
    "Settings",
    "UpdatePrompt",
    "ReplacePrompt",
    "UpdateSpeak",
    "UpdateThink",
    "UpdateListen",
    "InjectUserMessage",
    "InjectAgentMessage",
    "FunctionCallResponse",
    "ForceEndTurn",
    "KeepAlive",
}

# How long a session may go without any client message (audio or KeepAlive).
CLIENT_MESSAGE_TIMEOUT_SECS = 10.0
# How long a v1 turn ended by an interrupt waits for its last transcript.
V1_FINALIZE_TIMEOUT_SECS = 2.0


def error_message(code: str, description: str, *, warning: bool = False) -> dict[str, Any]:
    """Build a Voice Agent ``Error`` (or ``Warning``) message.

    Args:
        code: The error code.
        description: A human-readable description.
        warning: Whether to build a ``Warning`` instead of an ``Error``.

    Returns:
        The message.
    """
    return {"type": "Warning" if warning else "Error", "code": code, "description": description}


class DeepgramVoiceAgentProcessor(FrameProcessor):
    """Handles Voice Agent API client messages and sends server messages.

    Place it right after the transport's input, behind a transport using
    :class:`~pipecat.extensions.deepgram_voice_agent.serializer.DeepgramVoiceAgentSerializer`.
    It turns the client's control messages into frames:

    - ``UpdatePrompt`` / ``ReplacePrompt``: update the LLM's system instruction.
    - ``UpdateSpeak`` / ``UpdateThink``: update the TTS voice / LLM model.
    - ``InjectUserMessage``: add a user message and run the LLM.
    - ``InjectAgentMessage``: speak a message as the agent.
    - ``FunctionCallResponse``: answer a client function call with a
      :class:`~pipecat.frames.frames.ClientFunctionCallResultFrame`.

    A second ``Settings`` or a message it can't parse is a protocol error: it
    reports it and ends the session, as the Voice Agent API does. Errors from
    the pipeline are forwarded to the client.

    Messages for the client go out with :meth:`send`, which
    :class:`DeepgramVoiceAgentObserver` and server-side function handlers use.
    """

    def __init__(self, *, system_instruction: str = "", greeting: str | None = None, **kwargs):
        """Initialize the processor.

        Args:
            system_instruction: The prompt from the client's ``Settings``, which
                ``UpdatePrompt`` appends to.
            greeting: Text for the agent to speak when the session starts.
            **kwargs: Additional arguments passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)
        self._system_instruction = system_instruction
        self._greeting = greeting
        self._closing = False
        self._user_speaking = False
        self._agent_speaking = False
        # Set by an injected user message, so the observer can time the reply.
        self.on_turn_end: Callable[[], None] | None = None
        # Handles UpdateListen; without one, UpdateListen is refused.
        self.listen_updater: DeepgramListenUpdater | None = None
        # The listen services, for ForceEndTurn; the TTS, to tell its errors apart.
        self.listen: DeepgramListen | None = None
        self.tts: FrameProcessor | None = None
        # Checks an UpdateThink before it's applied, raising SettingsError.
        self.validate_think: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None
        # The end of the turn the client forced, for EndOfTurn's trigger.
        self.end_turn_forced = False
        # Agent messages waiting for the agent to finish what it is saying.
        self._queued_messages: deque[str] = deque()
        # A message to speak once the user turn it ended has been reported.
        self._speech_after_user_turn: str | None = None
        # A v1 turn being ended by an interrupt, waiting for its last words.
        self._v1_turn_ending = False
        self._v1_turn_timeout_task: asyncio.Task | None = None
        # Where agent speech enters (past the LLM), and the gate before the LLM.
        self.speech_injector: FrameProcessor = self
        self.llm_input_gate = DeepgramVoiceAgentLLMInputGate()
        # From the observer: whether the agent is replying or speaking.
        self.agent_busy: Callable[[], bool] = lambda: self._agent_speaking
        self._last_client_message = time.monotonic()
        self._client_timeout_task: asyncio.Task | None = None

    async def send(self, message: dict[str, Any]):
        """Send a message to the client.

        Args:
            message: The JSON-serializable message.
        """
        await self.push_frame(OutputTransportMessageUrgentFrame(message=message))

    async def send_conversation_text(self, role: str, content: str, **fields: Any):
        """Send a ``ConversationText`` and the ``History`` entry that goes with it.

        Args:
            role: ``user`` or ``assistant``.
            content: The text.
            **fields: Additional ``ConversationText`` fields.
        """
        await self.send({"type": "ConversationText", "role": role, "content": content, **fields})
        await self.send({"type": "History", "role": role, "content": content})

    async def close_with_error(self, code: str, description: str):
        """Report a protocol error to the client and end the session.

        Args:
            code: The error code.
            description: A human-readable description.
        """
        if self._closing:
            return
        self._closing = True
        logger.warning(f"{self} closing session: {code}: {description}")
        await self.send(error_message(code, description))
        await self.push_frame(EndWorkerFrame(reason=code), FrameDirection.UPSTREAM)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InputTransportMessageFrame):
            if direction == FrameDirection.DOWNSTREAM:
                self._last_client_message = time.monotonic()
                await self._handle_client_message(frame.message)
            return
        if isinstance(frame, InputAudioRawFrame):
            self._last_client_message = time.monotonic()

        await self.push_frame(frame, direction)

        if direction == FrameDirection.UPSTREAM:
            # Turn and speaking frames are broadcast, so their upstream copies
            # reach this processor at the head of the pipeline.
            if isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
                self._user_speaking = isinstance(frame, UserStartedSpeakingFrame)
                if self._user_speaking:
                    # Barging in drops what the agent had queued.
                    self._queued_messages.clear()
                if self.listen_updater:
                    await self.listen_updater.set_user_speaking(self._user_speaking)
            elif isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
                self._agent_speaking = isinstance(frame, BotStartedSpeakingFrame)

        if isinstance(frame, StartFrame):
            self._last_client_message = time.monotonic()
            self._client_timeout_task = self.create_task(self._watch_client_messages())
            if self._greeting:
                await self._speak_as_agent(self._greeting)
        elif isinstance(frame, (EndFrame, CancelFrame)) and self._client_timeout_task:
            await self.cancel_task(self._client_timeout_task)
            self._client_timeout_task = None
        elif isinstance(frame, ErrorFrame) and direction == FrameDirection.UPSTREAM:
            fatal = frame.fatal or (frame.processor is not None and not frame.processor.is_usable)
            if self.tts is not None and frame.processor is self.tts:
                await self.send(error_message(SPEAK_REQUEST_FAILED, str(frame.error), warning=True))
            else:
                await self.send(error_message(PIPELINE_ERROR, str(frame.error), warning=not fatal))

    async def _watch_client_messages(self):
        """End the session when the client goes quiet, as the Voice Agent API does."""
        while True:
            await asyncio.sleep(1)
            if time.monotonic() - self._last_client_message > CLIENT_MESSAGE_TIMEOUT_SECS:
                await self.close_with_error(
                    CLIENT_MESSAGE_TIMEOUT,
                    f"No message from the client for {CLIENT_MESSAGE_TIMEOUT_SECS:.0f} seconds; "
                    "send audio or KeepAlive messages.",
                )
                return

    async def _handle_client_message(self, message: Any):
        if not isinstance(message, dict) or message.get("type") not in CLIENT_MESSAGE_TYPES:
            await self.close_with_error(
                UNPARSABLE_CLIENT_MESSAGE, f"Could not parse client message: {message!r}"
            )
            return

        match message["type"]:
            case "Settings":
                await self.close_with_error(
                    SETTINGS_ALREADY_APPLIED, "Settings can only be sent once per session."
                )
            case "UpdatePrompt":
                self._system_instruction = f"{self._system_instruction}\n{message['prompt']}"
                await self._update_system_instruction()
                await self.send({"type": "PromptUpdated"})
            case "ReplacePrompt":
                self._system_instruction = message["prompt"]
                await self._update_system_instruction()
            case "UpdateSpeak":
                await self._update_speak(message.get("speak"))
            case "UpdateThink":
                await self._update_think(message.get("think"))
            case "UpdateListen":
                if self.listen_updater:
                    await self.listen_updater.update(message.get("listen"))
                else:
                    await self.send(
                        error_message(
                            UNSUPPORTED_UPDATE, "UpdateListen is not supported.", warning=True
                        )
                    )
            case "InjectUserMessage":
                await self._inject_user_message(message["content"])
            case "InjectAgentMessage":
                await self._inject_agent_message(
                    message.get("content", message.get("message", "")), message.get("behavior")
                )
            case "FunctionCallResponse":
                await self.push_frame(
                    ClientFunctionCallResultFrame(
                        tool_call_id=message["id"], result=message.get("content", "")
                    )
                )
            case "ForceEndTurn":
                await self._force_end_turn()
            case "KeepAlive":
                pass

    async def _update_system_instruction(self):
        await self.push_frame(
            LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=self._system_instruction))
        )

    async def _update_speak(self, speak: Any):
        try:
            delta = speak_settings(speak)
        except SettingsError as e:
            # Invalid speak settings end the session, as in the Voice Agent API.
            await self.close_with_error(e.code, e.description)
            return
        await self.push_frame(TTSUpdateSettingsFrame(delta=delta))
        await self.send({"type": "SpeakUpdated"})

    async def _update_think(self, think: Any):
        if self.validate_think:
            try:
                await self.validate_think(think or {})
            except SettingsError as e:
                await self.send(error_message(e.code, e.description))
                return
        model = (think or {}).get("provider", {}).get("model")
        if model:
            await self.push_frame(LLMUpdateSettingsFrame(delta=LLMSettings(model=model)))
        if think and "prompt" in think:
            self._system_instruction = think["prompt"]
            await self._update_system_instruction()
        await self.send({"type": "ThinkUpdated"})

    async def _force_end_turn(self):
        listen = self.listen
        if listen is None or listen.active is not listen.v2:
            await self.send(
                error_message(
                    FORCE_END_TURN_UNSUPPORTED,
                    "ForceEndTurn requires a Deepgram v2 (Flux) listen provider.",
                    warning=True,
                )
            )
            return
        self.end_turn_forced = True
        await listen.v2.force_end_turn()

    async def _inject_user_message(self, content: str):
        if self.on_turn_end:
            self.on_turn_end()
        # An injected message is a new user turn: it barges in on whatever the
        # agent is saying or about to say, which ends the agent's turn.
        was_speaking = self._agent_speaking
        self._queued_messages.clear()
        self.llm_input_gate.skip_next_inference = False
        await self.send({"type": "UserStartedSpeaking"})
        await self.broadcast_interruption()
        if was_speaking:
            await self.send({"type": "AgentAudioDone"})
        await self.send_conversation_text("user", content)
        await self.push_frame(
            LLMMessagesAppendFrame([{"role": "user", "content": content}], run_llm=True)
        )

    async def _inject_agent_message(self, content: str, behavior: str | None):
        """Speak a message as the agent, by the Voice Agent API's rules.

        Without a ``behavior`` the message is refused while the user or the
        agent is speaking. ``queue`` is refused while the user speaks, and
        otherwise waits for the agent to finish what it is saying (a barge-in
        drops it). ``interrupt`` cuts the agent off, ahead of any queued
        messages, and ends a user turn in progress: what the user has said so far becomes their turn, with no
        reply, and the message is spoken.
        """
        if behavior == "interrupt":
            # Jumps ahead of queued messages, which follow it; only the user
            # barging in drops them.
            if self._user_speaking:
                await self._end_user_turn_then_speak(content)
                return
            # Interrupting from where the speech enters keeps the interruption
            # ahead of the speech on its way to the TTS.
            await self.speech_injector.broadcast_interruption()
            await self._speak_as_agent(content)
            return
        if self._user_speaking:
            await self.send(
                error_message(
                    INJECT_AGENT_MESSAGE_DURING_USER_SPEECH,
                    "The agent can't speak while the user is speaking.",
                    warning=True,
                )
            )
            return
        if behavior is None and self._agent_speaking:
            await self.send(
                error_message(
                    INJECT_AGENT_MESSAGE_DURING_AGENT_SPEECH,
                    "The agent is already speaking.",
                    warning=True,
                )
            )
            return
        if behavior == "queue" and self.agent_busy():
            self._queued_messages.append(content)
            return
        await self._speak_as_agent(content)

    async def _end_user_turn_then_speak(self, content: str):
        """End the user's turn where it is, with no reply, then speak as the agent."""
        # The turn's inference is dropped: the user didn't finish, so there is
        # nothing to answer yet.
        self.llm_input_gate.skip_next_inference = True
        listen = self.listen
        if listen is not None and listen.active is listen.v2:
            # Flux closes the turn with what it heard so far, which is then
            # reported and recorded as usual; the message follows it.
            self._speech_after_user_turn = content
            self.end_turn_forced = True
            await listen.v2.force_end_turn()
            return
        # A v1 model can't end a turn on request, but it can transcribe all the
        # audio sent so far; once that's in, the turn is ended there, the way
        # Flux ends one, and reported as usual.
        self._speech_after_user_turn = content
        self._v1_turn_ending = True
        if listen is not None:
            await listen.v1.finalize()
        self._v1_turn_timeout_task = self.create_task(self._end_v1_turn_soon())

    async def _end_v1_turn_soon(self):
        await asyncio.sleep(V1_FINALIZE_TIMEOUT_SECS)
        self._v1_turn_timeout_task = None
        await self.end_v1_turn()

    async def end_v1_turn(self):
        """End a v1 user turn that ``interrupt`` cut off, once its words are in.

        Proposes the end of the turn, as Flux does; the v1 service's turn
        strategies take it, and the turn is reported and recorded as usual.
        """
        if not self._v1_turn_ending:
            return
        self._v1_turn_ending = False
        if self._v1_turn_timeout_task:
            await self.cancel_task(self._v1_turn_timeout_task)
            self._v1_turn_timeout_task = None
        await self.push_frame(ProposedUserStoppedSpeakingFrame())

    async def on_user_turn_reported(self):
        """Speak a message that was waiting for the user's turn to be reported."""
        content, self._speech_after_user_turn = self._speech_after_user_turn, None
        if content:
            await self._speak_as_agent(content)

    async def release_queued_messages(self) -> bool:
        """Speak the queued agent messages, now that the agent has nothing else to say.

        Returns:
            Whether there were any.
        """
        if not self._queued_messages:
            return False
        messages = list(self._queued_messages)
        self._queued_messages.clear()
        for content in messages:
            await self._speak_as_agent(content)
        return True

    async def _speak_as_agent(self, text: str):
        # Pushed past the LLM, which holds frames behind a response in progress.
        # The observer reports the text once its audio starts.
        await self.speech_injector.push_frame(TTSSpeakFrame(text))


class DeepgramVoiceAgentLLMInputGate(FrameProcessor):
    """Drops one inference when told to, for a user turn that gets no reply.

    Goes right before the LLM. Every other frame passes through.
    """

    def __init__(self, **kwargs):
        """Initialize the gate.

        Args:
            **kwargs: Additional arguments passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)
        self.skip_next_inference = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Drop the next inference if asked to; pass everything else on.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and self.skip_next_inference:
            if isinstance(frame, LLMContextFrame):
                self.skip_next_inference = False
                return
            if isinstance(frame, UserStartedSpeakingFrame):
                # A new turn; the one whose inference was to be dropped had none.
                self.skip_next_inference = False
        await self.push_frame(frame, direction)


class DeepgramVoiceAgentSpeechInjector(FrameProcessor):
    """Where agent speech enters the pipeline, past the LLM.

    Goes between the LLM and the TTS, so speech injected while the LLM is busy
    doesn't wait behind it. Every frame passes through.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Pass frames on.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class DeepgramVoiceAgentObserver(BaseObserver):
    """Reports pipeline activity to the client as Voice Agent API events.

    Sends, through a :class:`DeepgramVoiceAgentProcessor`:

    - ``UserStartedSpeaking`` when a user turn starts.
    - ``ConversationText`` (``user``) when a user turn ends, with the turn's
      final transcripts and, from Flux, the detected ``languages``.
    - ``ConversationText`` (``assistant``) for each spoken segment, right after
      its first audio reaches the client, as in the Voice Agent API. A segment ends where its ``TTSTextFrame`` follows its
      audio out of the transport, so this needs a TTS service that reports text
      per segment (one without word timestamps).
    - A ``History`` entry after each ``ConversationText``.
    - ``FunctionCallRequest`` for each client function call.
    - ``FunctionCallCancelled`` when a call the client holds is cancelled.
    - ``EndOfTurn`` with its ``trigger`` (``model``, ``timeout`` or ``manual``)
      when Flux ends a user turn.
    - ``AgentStartedSpeaking`` with latencies, after a turn's first
      ``ConversationText`` (only for an ``experimental`` session, as in the Voice
      Agent API), and ``AgentAudioDone``
      once the agent has nothing left to say, or right after
      ``UserStartedSpeaking`` when the user barges in.
    """

    def __init__(
        self, processor: DeepgramVoiceAgentProcessor, *, experimental: bool = False, **kwargs
    ):
        """Initialize the observer.

        Args:
            processor: The processor that sends messages to the client.
            experimental: The ``Settings`` ``experimental`` flag. The Voice Agent
                API sends ``AgentStartedSpeaking`` only when it is set.
            **kwargs: Additional arguments passed to :class:`BaseObserver`.
        """
        super().__init__(**kwargs)
        self._processor = processor
        self._processor.on_turn_end = self._mark_turn_end
        self._experimental = experimental
        self._response_text = ""
        self._response_called_functions = False
        # Audio cut short by an interruption doesn't end with AgentAudioDone.
        self._interrupted = False
        # AgentStartedSpeaking goes out once per agent turn.
        self._speaking_announced = False
        self._turn_end_time: float | None = None
        self._first_text_time: float | None = None
        # The user turn in progress: its final transcripts, the latest interim
        # one, and the detected languages.
        self._turn_transcripts: list[str] = []
        self._turn_interim = ""
        self._turn_languages: list[str] | None = None
        # Returns the listen language hints in effect, for ``languages_hinted``.
        self.languages_hinted: Callable[[], list[str] | None] = lambda: None
        # The text of the segment whose audio the transport is about to send,
        # announced with that audio's first chunk.
        self._next_segment_text: str | None = None
        # Segments the TTS has started that haven't finished reaching the client.
        self._open_segments = 0
        self._bot_speaking = False
        self._response_in_progress = False
        # AgentAudioDone held back until the agent has nothing left to say.
        self._audio_done_pending = False
        # Client function calls sent and not yet settled: tool call id to name.
        self._client_calls: dict[str, str] = {}

    def _mark_turn_end(self):
        self._turn_end_time = time.monotonic()
        self._first_text_time = None

    def agent_busy(self) -> bool:
        """Whether the agent is replying or speaking.

        Returns:
            Whether it is.
        """
        return self._response_in_progress or self._bot_speaking or self._open_segments > 0

    async def _send_user_text(self):
        content = " ".join(self._turn_transcripts).strip()
        self._turn_transcripts = []
        self._turn_interim = ""
        if not content:
            await self._processor.on_user_turn_reported()
            return
        fields: dict[str, Any] = {}
        if self._turn_languages is not None:
            fields["languages"] = self._turn_languages
            fields["languages_hinted"] = self.languages_hinted()
        await self._processor.send_conversation_text("user", content, **fields)
        await self._processor.on_user_turn_reported()

    async def _on_output(self, frame: Frame):
        """Follow spoken segments as the output transport sends them on.

        A TTS service pushes each segment's text (an ``AggregatedTextFrame``
        that will be spoken) ahead of its audio, and a ``TTSTextFrame`` after
        it; the transport sends them on in that order, and drops what an
        interruption cuts. So the text is announced with the segment's first
        audio, as in the Voice Agent API.
        """
        if isinstance(frame, TTSTextFrame):
            self._open_segments = max(0, self._open_segments - 1)
            await self._maybe_send_audio_done()
        elif isinstance(frame, AggregatedTextFrame):
            if frame.will_be_spoken and frame.text.strip():
                self._next_segment_text = frame.text.strip()
        elif isinstance(frame, OutputAudioRawFrame) and self._next_segment_text:
            text, self._next_segment_text = self._next_segment_text, None
            await self._processor.send_conversation_text("assistant", text)
            if not self._speaking_announced:
                self._speaking_announced = True
                if self._experimental:
                    await self._processor.send(
                        {"type": "AgentStartedSpeaking", **self._latencies()}
                    )
        elif isinstance(frame, InterruptionFrame):
            self._next_segment_text = None
            self._open_segments = 0

    async def _on_transcription(self, frame: InterimTranscriptionFrame | TranscriptionFrame):
        """Collect the user turn's text; report the end of a Flux turn."""
        if isinstance(frame, InterimTranscriptionFrame):
            self._turn_interim = frame.text
            return
        self._turn_transcripts.append(frame.text)
        self._turn_interim = ""
        result = frame.result if isinstance(frame.result, dict) else {}
        if getattr(frame.result, "from_finalize", False):
            # Every word heard before a v1 finalize is in.
            await self._processor.end_v1_turn()
        if result.get("languages"):
            self._turn_languages = list(result["languages"])
        if result.get("event") == "EndOfTurn":
            # Flux names what ended the turn.
            forced, self._processor.end_turn_forced = self._processor.end_turn_forced, False
            trigger = result.get("trigger") or ("manual" if forced else "model")
            await self._processor.send({"type": "EndOfTurn", "trigger": trigger})

    async def _maybe_send_audio_done(self):
        """Send a held AgentAudioDone once the agent has nothing left to say."""
        if not self._audio_done_pending or self._bot_speaking:
            return
        if self._open_segments or self._response_in_progress:
            return
        if await self._processor.release_queued_messages():
            # Queued messages are spoken first; AgentAudioDone follows them.
            return
        self._audio_done_pending = False
        await self._processor.send({"type": "AgentAudioDone"})

    async def on_push_frame(self, data: FramePushed):
        """Translate pipeline frames into Voice Agent events.

        Args:
            data: The frame push event.
        """
        if data.direction != FrameDirection.DOWNSTREAM:
            return
        if isinstance(data.source, BaseOutputTransport):
            await self._on_output(data.frame)
        frame = data.frame
        send = self._processor.send
        if not data.first_push:
            return
        if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            await self._on_transcription(frame)
            return

        if (
            isinstance(frame, AggregatedTextFrame)
            and not isinstance(frame, TTSTextFrame)
            and frame.will_be_spoken
        ):
            # The TTS starting a segment.
            self._open_segments += 1
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._turn_transcripts = []
            self._turn_languages = None
            await send({"type": "UserStartedSpeaking"})
            if self._bot_speaking:
                # Barging in ends the agent's turn; its cut-off audio sends
                # no AgentAudioDone of its own.
                await send({"type": "AgentAudioDone"})
        elif isinstance(frame, InterruptionFrame):
            self._interrupted = True
            self._speaking_announced = False
            self._response_in_progress = False
            self._audio_done_pending = False
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._mark_turn_end()
            await self._send_user_text()
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._response_text = ""
            self._response_called_functions = False
            self._response_in_progress = True
        elif isinstance(frame, LLMTextFrame):
            if self._first_text_time is None:
                self._first_text_time = time.monotonic()
            self._response_text += frame.text
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._response_called_functions = True
        elif isinstance(frame, LLMFullResponseEndFrame):
            text = self._response_text.strip()
            self._response_text = ""
            self._response_in_progress = False
            if not text and not self._response_called_functions:
                # Nothing will be spoken, so no audio will end.
                await send({"type": "AgentAudioDone"})
            else:
                await self._maybe_send_audio_done()
        elif isinstance(frame, FunctionCallCancelFrame):
            name = self._client_calls.pop(frame.tool_call_id, None)
            if name is not None:
                # The client holds this call; tell it the call is off.
                await send(
                    {
                        "type": "FunctionCallCancelled",
                        "functions": [{"id": frame.tool_call_id, "name": name}],
                    }
                )
        elif isinstance(frame, FunctionCallResultFrame):
            self._client_calls.pop(frame.tool_call_id, None)
        elif isinstance(frame, ClientFunctionCallRequestFrame):
            self._client_calls[frame.tool_call_id] = frame.function_name
            await send(
                {
                    "type": "FunctionCallRequest",
                    "functions": [
                        {
                            "id": frame.tool_call_id,
                            "name": frame.function_name,
                            "arguments": json.dumps(dict(frame.arguments)),
                            "client_side": True,
                        }
                    ],
                }
            )
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._interrupted = False
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._speaking_announced = False
            if not self._interrupted:
                # Held until any segments still queued have been spoken too.
                self._audio_done_pending = True
                await self._maybe_send_audio_done()

    def _latencies(self) -> dict[str, float]:
        now = time.monotonic()
        turn_end = self._turn_end_time or now
        first_text = self._first_text_time or now
        return {
            "total_latency": max(0.0, now - turn_end),
            "ttt_latency": max(0.0, first_text - turn_end),
            "tts_latency": max(0.0, now - first_text),
        }
