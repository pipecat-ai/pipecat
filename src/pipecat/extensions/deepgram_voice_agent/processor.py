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

import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ClientFunctionCallRequestFrame,
    ClientFunctionCallResultFrame,
    EndWorkerFrame,
    ErrorFrame,
    Frame,
    FunctionCallsStartedFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
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
from pipecat.services.settings import LLMSettings, TTSSettings
from pipecat.transports.base_output import BaseOutputTransport

if TYPE_CHECKING:
    from pipecat.extensions.deepgram_voice_agent.listen import DeepgramListenUpdater

# Error codes the Voice Agent API reports.
NON_SETTINGS_MESSAGE_BEFORE_SETTINGS = "NON_SETTINGS_MESSAGE_BEFORE_SETTINGS"
SETTINGS_ALREADY_APPLIED = "SETTINGS_ALREADY_APPLIED"
UNPARSABLE_CLIENT_MESSAGE = "UNPARSABLE_CLIENT_MESSAGE"
INVALID_SETTINGS = "INVALID_SETTINGS"
UNSUPPORTED_UPDATE = "UNSUPPORTED_UPDATE"
PIPELINE_ERROR = "PIPELINE_ERROR"
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
    "KeepAlive",
}


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
                await self._handle_client_message(frame.message)
            return

        await self.push_frame(frame, direction)

        if direction == FrameDirection.UPSTREAM:
            # Turn and speaking frames are broadcast, so their upstream copies
            # reach this processor at the head of the pipeline.
            if isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
                self._user_speaking = isinstance(frame, UserStartedSpeakingFrame)
                if self.listen_updater:
                    await self.listen_updater.set_user_speaking(self._user_speaking)
            elif isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
                self._agent_speaking = isinstance(frame, BotStartedSpeakingFrame)

        if isinstance(frame, StartFrame) and self._greeting:
            await self._speak_as_agent(self._greeting)
        elif isinstance(frame, ErrorFrame) and direction == FrameDirection.UPSTREAM:
            fatal = frame.fatal or (frame.processor is not None and not frame.processor.is_usable)
            await self.send(error_message(PIPELINE_ERROR, str(frame.error), warning=not fatal))

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
            case "KeepAlive":
                pass

    async def _update_system_instruction(self):
        await self.push_frame(
            LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=self._system_instruction))
        )

    async def _update_speak(self, speak: Any):
        speak = speak[0] if isinstance(speak, list) and speak else speak
        model = (speak or {}).get("provider", {}).get("model")
        if not model:
            await self.send(
                error_message(INVALID_SETTINGS, "UpdateSpeak needs speak.provider.model.")
            )
            return
        await self.push_frame(TTSUpdateSettingsFrame(delta=TTSSettings(voice=model)))
        await self.send({"type": "SpeakUpdated"})

    async def _update_think(self, think: Any):
        model = (think or {}).get("provider", {}).get("model")
        if model:
            await self.push_frame(LLMUpdateSettingsFrame(delta=LLMSettings(model=model)))
        if think and "prompt" in think:
            self._system_instruction = think["prompt"]
            await self._update_system_instruction()
        await self.send({"type": "ThinkUpdated"})

    async def _inject_user_message(self, content: str):
        if self.on_turn_end:
            self.on_turn_end()
        # An injected message is a new user turn: it barges in on whatever the
        # agent is saying or about to say, which ends the agent's turn.
        was_speaking = self._agent_speaking
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

        Without a ``behavior`` the message is refused while anyone is speaking.
        ``queue`` waits for the agent's current speech (a barge-in drops it) and
        is refused while the user speaks. ``interrupt`` cuts the agent off.
        """
        if self._user_speaking and behavior != "interrupt":
            await self.send(
                error_message(
                    INJECT_AGENT_MESSAGE_DURING_USER_SPEECH,
                    "The agent can't speak while the user is speaking.",
                    warning=True,
                )
            )
            return
        if self._agent_speaking and behavior is None:
            await self.send(
                error_message(
                    INJECT_AGENT_MESSAGE_DURING_AGENT_SPEECH,
                    "The agent is already speaking.",
                    warning=True,
                )
            )
            return
        if behavior == "interrupt":
            await self.broadcast_interruption()
        await self._speak_as_agent(content)

    async def _speak_as_agent(self, text: str):
        # The observer reports the text once its audio starts.
        await self.push_frame(TTSSpeakFrame(text))


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
        # The user turn in progress: its final transcripts and detected languages.
        self._turn_transcripts: list[str] = []
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

    def _mark_turn_end(self):
        self._turn_end_time = time.monotonic()
        self._first_text_time = None

    async def _send_user_text(self):
        content = " ".join(self._turn_transcripts).strip()
        self._turn_transcripts = []
        if not content:
            return
        fields: dict[str, Any] = {}
        if self._turn_languages is not None:
            fields["languages"] = self._turn_languages
            fields["languages_hinted"] = self.languages_hinted()
        await self._processor.send_conversation_text("user", content, **fields)

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

    async def _maybe_send_audio_done(self):
        """Send a held AgentAudioDone once the agent has nothing left to say."""
        if not self._audio_done_pending or self._bot_speaking:
            return
        if self._open_segments or self._response_in_progress:
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
        if not data.first_push:
            return
        frame = data.frame
        send = self._processor.send

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
        elif isinstance(frame, TranscriptionFrame):
            self._turn_transcripts.append(frame.text)
            result = frame.result if isinstance(frame.result, dict) else {}
            if result.get("languages"):
                self._turn_languages = list(result["languages"])
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
        elif isinstance(frame, ClientFunctionCallRequestFrame):
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
