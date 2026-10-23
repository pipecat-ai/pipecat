#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Zoom Voice Agent speech-to-speech service.

The Zoom Voice Agent runs the whole voice loop on Zoom's side: turn detection,
speech recognition, the model and speech synthesis. Its events follow the
OpenAI Realtime protocol, so :class:`ZoomAIVoiceAgentLLMService` builds on
:class:`~pipecat.services.openai.realtime.llm.OpenAIRealtimeLLMService` for
interruptions, function calling and transcripts. The session configuration is
Zoom's own (see :mod:`pipecat.services.zoom_ai.voice_agent.events`).

See https://developers.zoom.us/docs/ai-services/voice-agent/live-mode/
"""

import asyncio
import base64
import copy
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from loguru import logger
from pydantic import BaseModel
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.typing import Subprotocol

from pipecat import version as pipecat_version
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import LLMFullResponseEndFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.openai.realtime import events as openai_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

from . import events

# The OpenAI Realtime service plays agent audio at 24 kHz; the Voice Agent speaks at 16 kHz.
_PLAYBACK_SAMPLE_RATE = 24000
_VOICE_AGENT_OUTPUT_RATE = 16000

# Non-terminal errors that are expected during normal turn-taking (an interruption
# cancelling a response that already finished, a response requested while another
# runs).
_EXPECTED_ERRORS = ("no active response", "already has an active response")

# Client messages must stay small; audio is appended in chunks well under the limit.
_MAX_APPEND_BYTES = 16384

# Caller audio that arrives before the session is ready (while connecting or
# reconnecting) is held and sent once it is, up to this much of the latest audio.
_PENDING_AUDIO_SECS = 3.0

# Reconnect attempts, in a row, after a session ends in a way that can be retried
# (the connection closing, or a session reaching its time limit); the count starts
# again once a new session has answered.
_MAX_RECONNECTS = 3
_RECONNECT_BACKOFF_SECS = 0.5  # doubled for each attempt in a row


@dataclass
class ZoomAIVoiceAgentLLMSettings(OpenAIRealtimeLLMService.Settings):
    """Settings for :class:`ZoomAIVoiceAgentLLMService`.

    ``model``, ``system_instruction`` and the ``session_properties`` fields for
    tools, ``tool_choice``, ``max_output_tokens`` and ``reasoning`` configure the
    model (``think``), as for other realtime services.

    Parameters:
        listen: How the agent listens: the caller's language and vocabulary.
            Fixed once the session is configured. ``None`` takes the server's
            defaults.
        speak: How the agent speaks: its voice. ``None`` takes the server's
            default voice.

    ``extra`` holds Voice Agent session fields this class has no setting for,
    shaped like the ``session`` of ``session.update`` (e.g.
    ``{"think": {"provider": {"type": "openai", ...}}}``). It is merged into each
    session update over the settings above.
    """

    listen: events.Listen | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speak: events.Speak | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    def __post_init__(self):
        # Accept the settings as plain dicts too, e.g. a Voice Agent session's JSON.
        if isinstance(self.listen, Mapping):
            self.listen = events.Listen.model_validate(self.listen)
        if isinstance(self.speak, Mapping):
            self.speak = events.Speak.model_validate(self.speak)


def _dump(value: Any) -> Any:
    return value.model_dump(exclude_none=True) if isinstance(value, BaseModel) else value


def _merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with ``extra`` merged in, nested objects merged key by key; ``extra`` wins."""
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _tracking_id() -> str:
    """A new ``x-zm-trackingid``, which Zoom support uses to find a session."""
    return f"zoom_ai_community_pipecat_{uuid.uuid4()}"


def _describe_refusal(exception: Exception) -> str | None:
    """Describe Zoom refusing the handshake (a bad API key, say), if that is what failed."""
    if isinstance(exception, InvalidStatus):
        response = cast(InvalidStatus, exception).response
        try:
            reason = json.loads(response.body).get("message")
        except (TypeError, ValueError, AttributeError):  # no body, or not a JSON object
            reason = None
        if reason:
            return f"Zoom refused the connection (HTTP {response.status_code}): {reason}"
    return None


class _ZoomAIVoiceAgentWebSocket:
    """The Voice Agent connection, presented as the OpenAI Realtime service expects.

    Incoming events are passed on unchanged except where the Voice Agent differs
    from what the OpenAI Realtime event models accept:

    - An event wrapped as ``{"type", "payload"}`` is unwrapped.
    - ``conversation.item.created`` (its acknowledgement of a client-created
      item) is delivered as ``conversation.item.added``.
    - Fields those models require are filled in when absent (``event_id``,
      ``response.status_details``, the usage details, ``error.type``).
    - Event types the models don't know are ignored.
    - Non-terminal errors go to ``on_error`` instead of ending the session; only
      errors marked ``fatal``, and failed ``conversation.item.retrieve`` requests
      (which the OpenAI service settles itself), are passed on.
    - Agent audio is resampled to 24 kHz as one stream per reply: a reply's audio
      arrives in bursts with gaps of up to half a second, so the resampler is never
      cleared between deltas (that would drop audio and restart its filter mid-word,
      a click), its tail is flushed into the reply's last delta, and it starts afresh
      with each reply.
    """

    def __init__(
        self, websocket, *, on_error: Callable[[dict[str, Any], set[str]], Awaitable[None]]
    ):
        self._ws = websocket
        self._on_error = on_error
        # The terminal error that ended the session, if one did.
        self.terminal_error: dict[str, Any] | None = None
        self._output_rate = _VOICE_AGENT_OUTPUT_RATE
        # One stream per reply: never cleared on a gap between deltas (see above).
        self._resampler = create_stream_resampler(clear_after_secs=None)
        # A delta can end mid-sample (an odd number of PCM16 bytes): the odd byte is kept
        # for the next one.
        self._partial_sample = b""

    async def send(self, message: str) -> None:
        await self._ws.send(message)

    async def close(self) -> None:
        await self._ws.close()

    def __aiter__(self):
        return self._messages()

    async def _messages(self):
        async for raw in self._ws:
            if isinstance(raw, bytes):
                continue  # the protocol is JSON text frames only
            for message in await self._adapt(raw):
                yield message

    async def _adapt(self, raw: str) -> list[str]:
        """The OpenAI-shaped messages a Voice Agent message becomes: usually one, none to skip it."""
        event = json.loads(raw)
        if isinstance(event.get("payload"), dict):
            event = event["payload"]
        kind = event.get("type", "")
        if kind == "conversation.item.created":
            kind = event["type"] = "conversation.item.added"
        if kind not in openai_events._server_event_types:
            logger.trace(f"Zoom Voice Agent: ignoring {kind!r} event")
            return []
        event.setdefault("event_id", "")

        if kind == "error":
            error = event["error"] = event.get("error") or {}
            error.setdefault("type", "server_error")
            error.setdefault("message", str(error.get("code") or "Zoom Voice Agent error"))
            if error.get("fatal"):
                self.terminal_error = error
            elif str(error.get("event_id") or "").startswith("rci_"):
                # The error correlates to retrieve_conversation_item(), whose
                # caller is waiting for it.
                error["code"] = "item_retrieve_invalid_item_id"
            else:
                # The client event it answers may be named in the error or the event.
                ids = {str(i) for i in (error.get("event_id"), event.get("event_id")) if i}
                await self._on_error(error, ids)
                return []
        elif kind == "session.updated":
            speak = (event.get("session") or {}).get("speak") or {}
            rate = (speak.get("format") or {}).get("rate")
            if rate:
                self._output_rate = int(rate)
        elif kind in ("response.created", "response.done"):
            self._partial_sample = b""  # a reply's audio never continues into the next one
            if self._output_rate != _PLAYBACK_SAMPLE_RATE:
                await self._resampler.reset()  # nor does the resampler's history
            self._fill_response(event.get("response"), done=kind == "response.done")
        elif kind == "response.output_audio.done" and self._output_rate != _PLAYBACK_SAMPLE_RATE:
            # The resampler holds the last few milliseconds back until it's flushed.
            tail = await self._resampler.flush()
            if tail:
                delta = {
                    "type": "response.output_audio.delta",
                    "event_id": "",
                    "response_id": event.get("response_id", ""),
                    "item_id": event.get("item_id", ""),
                    "output_index": event.get("output_index", 0),
                    "content_index": event.get("content_index", 0),
                    "delta": base64.b64encode(tail).decode(),
                }
                return [json.dumps(delta), json.dumps(event)]
        elif kind == "response.output_audio.delta":
            audio = self._partial_sample + base64.b64decode(event.get("delta", ""))
            cut = len(audio) - len(audio) % 2
            audio, self._partial_sample = audio[:cut], audio[cut:]
            if self._output_rate != _PLAYBACK_SAMPLE_RATE and audio:
                audio = await self._resampler.resample(
                    audio, self._output_rate, _PLAYBACK_SAMPLE_RATE
                )
            event["delta"] = base64.b64encode(audio).decode()
        return [json.dumps(event)]

    @staticmethod
    def _fill_response(response: Any, *, done: bool) -> None:
        if not isinstance(response, dict):
            return
        response.setdefault("object", "realtime.response")
        response.setdefault("output", [])
        # The reply's audio settings are the Voice Agent's (PCM at 16 kHz), which the
        # OpenAI event model doesn't take; the audio itself is resampled to 24 kHz here.
        response.pop("audio", None)
        if response.get("status") == "failed" and not response.get("status_details"):
            response["status_details"] = {
                "type": "failed",
                "error": {"type": "server_error", "message": "Zoom Voice Agent response failed"},
            }
        response.setdefault("status_details", None)
        if done:
            usage: dict[str, Any] = response.get("usage") or {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
            usage.setdefault("input_token_details", {})
            usage.setdefault("output_token_details", {})
            response["usage"] = usage


class ZoomAIVoiceAgentLLMService(OpenAIRealtimeLLMService):
    """Zoom Voice Agent speech-to-speech service.

    Replaces speech-to-text, the LLM and text-to-speech in a pipeline: the
    caller's audio goes to the Voice Agent and the agent's audio comes back.
    Function calling, interruptions and transcripts work as for
    :class:`~pipecat.services.openai.realtime.llm.OpenAIRealtimeLLMService`.
    The Voice Agent detects turns and handles barge-in itself.

    Input audio is sent at the pipeline's input sample rate. Audio that arrives
    while the session is connecting or reconnecting is held (up to the latest
    three seconds) and sent once the session is ready, so the caller's first words
    aren't lost. The agent speaks at 16 kHz, played at 24 kHz. A function call's
    output is created right after the call in the conversation, where the Voice
    Agent's model reads it.

    The model and the listen settings are sent in the first session update
    only; later updates change the instructions, tools and voice. Errors the
    Voice Agent reports without ending the session are pushed as non-fatal
    errors. If the session ends unexpectedly (the connection closing, or a
    session reaching its time limit), the service reconnects and restores the
    conversation; otherwise it pushes a permanent error.

    Example::

        from pipecat.services.zoom_ai.voice_agent.events import Speak

        llm = ZoomAIVoiceAgentLLMService(
            api_key=os.getenv("ZOOM_AI_API_KEY"),
            settings=ZoomAIVoiceAgentLLMService.Settings(
                system_instruction="You are a helpful assistant.",
                speak=Speak(voice="jenna"),
            ),
        )
    """

    Settings = ZoomAIVoiceAgentLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.zoom.us/v2/ws/aiservices/voice-agent/live",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Zoom Voice Agent service.

        Args:
            api_key: Zoom AI Services API key. It is sent on every
                connection and reconnect, so it must stay valid for as long as
                the service runs.
            url: Voice Agent WebSocket URL.
            settings: Session settings. Defaults to the ``gemma-4`` model and the
                server's default voice.
            **kwargs: Additional arguments passed to the parent
                :class:`~pipecat.services.openai.realtime.llm.OpenAIRealtimeLLMService`.
        """
        default_settings = self.Settings(
            model="gemma-4",
            session_properties=openai_events.SessionProperties(),
            listen=None,
            speak=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(api_key=api_key, base_url=url, settings=default_settings, **kwargs)
        # The model is part of the session configuration, not the URL.
        self.base_url = url
        self._input_sample_rate = 16000
        # The model and listen settings go in the first session update of a
        # connection only; they can't change after that.
        self._listen_configured = False
        # call_id -> the item id of its function call in the conversation
        self._call_items: dict[str, str] = {}
        # event_id of a response.create the server hasn't answered yet
        self._pending_response_create: str | None = None
        self._terminal_message: str | None = None
        self._reconnects = 0
        self._restore_conversation_when_ready = False
        self._input_resampler = create_stream_resampler()
        # Caller audio, at the session's input rate, waiting for the session to be ready.
        self._pending_audio = bytearray()
        self._tracking_id: str | None = None  # the current session's x-zm-trackingid

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect, sending audio at the pipeline's input rate.

        Args:
            setup: Configuration object containing setup parameters.
        """
        self._input_sample_rate = setup.audio_in_sample_rate
        await super().setup(setup)

    def _is_turn_detection_disabled(self) -> bool:
        # The Voice Agent always detects turns itself.
        return False

    async def _connect(self):
        try:
            if self._websocket:
                return
            self._tracking_id = _tracking_id()
            logger.debug(f"Connecting to the Zoom Voice Agent (tracking_id={self._tracking_id})")
            websocket = await websocket_connect(
                uri=self.base_url,
                subprotocols=[Subprotocol("voice-agent.zoom.v2")],
                additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": f"pipecat/{pipecat_version()}",
                    "x-zm-trackingid": self._tracking_id,
                },
                max_size=None,
            )
            self._listen_configured = False
            self._call_items = {}
            self._pending_response_create = None
            self._terminal_message = None
            self._websocket = _ZoomAIVoiceAgentWebSocket(
                websocket, on_error=self._handle_server_error
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            self._websocket = None
            refusal = _describe_refusal(e)
            await self.push_error(
                error_msg=(refusal or f"Unable to connect to the Zoom Voice Agent: {e}")
                + f" (tracking_id={self._tracking_id})",
                exception=e,
            )

    async def _receive_task_handler(self):
        closed: ConnectionClosed | None = None
        try:
            await super()._receive_task_handler()
        except ConnectionClosed as e:
            closed = e
        if self._disconnecting or not self._websocket:
            return
        # The session is over: a terminal error, or the connection closed.
        websocket, self._websocket = self._websocket, None
        terminal = websocket.terminal_error
        interrupted_reply = self._response_active
        await self._end_session()
        try:
            await websocket.close()
        except Exception:
            pass

        reason = self._terminal_message or (str(closed) if closed else "connection closed")
        retryable = terminal is None or bool(terminal.get("retryable"))
        if retryable:
            logger.warning(f"{self}: Zoom Voice Agent session ended ({reason}); reconnecting")
            self._llm_needs_conversation_setup = True
            self._restore_conversation_when_ready = True
            # A reply the session ended in, or one waiting for it to be ready, is
            # asked for in the new one.
            self._run_llm_when_api_session_ready |= interrupted_reply
            while self._reconnects < _MAX_RECONNECTS and not self._disconnecting:
                self._reconnects += 1
                await asyncio.sleep(_RECONNECT_BACKOFF_SECS * 2 ** (self._reconnects - 1))
                await self._connect()
                if self._websocket:
                    return
            if self._disconnecting:
                return
            reason += "; reconnecting failed"
        await self.push_error(
            error_msg=f"Zoom Voice Agent session ended: {reason} (tracking_id={self._tracking_id})",
            force_treat_as_permanent=True,
        )

    async def _end_session(self):
        """Settle what the ended session left open."""
        self._api_session_ready = False
        if self._current_audio_response is not None:
            self._current_audio_response = None
            await self.push_frame(TTSStoppedFrame())
        if self._response_active or self._current_assistant_response:
            await self.push_frame(LLMFullResponseEndFrame())
        self._current_assistant_response = None
        await self.stop_all_metrics()
        # A new session knows none of the old call ids, and no response runs in it.
        self._open_function_calls = set()
        self._pending_function_calls = {}
        self._call_items = {}
        self._messages_added_manually = {}
        self._response_active = False
        self._response_create_owed = False
        self._pending_response_create = None
        futures, self._retrieve_conversation_item_futures = (
            self._retrieve_conversation_item_futures,
            {},
        )
        for waiting in futures.values():
            for future in waiting:
                if not future.done():
                    future.set_exception(Exception("Zoom Voice Agent session ended"))

    async def _handle_evt_error(self, evt):
        # A terminal error: the receive loop ends, and the session with it.
        self._terminal_message = evt.error.message
        logger.debug(f"{self}: terminal error: {evt.error.message}")

    async def _restore_conversation(self):
        """Recreate the conversation so far in a new session."""
        if not self._context or not self._llm_needs_conversation_setup:
            return
        params = await self.get_llm_adapter().get_llm_invocation_params(self._context)
        for item in params["messages"]:
            evt = openai_events.ConversationItemCreateEvent(item=item)
            self._messages_added_manually[evt.item.id] = True
            await self.send_client_event(evt)
        self._llm_needs_conversation_setup = False

    async def send_client_event(self, event: openai_events.ClientEvent):
        """Send a client event to the Voice Agent.

        Args:
            event: The client event to send.
        """
        if isinstance(event, openai_events.ResponseCreateEvent):
            self._pending_response_create = event.event_id
            # The Voice Agent's response.create takes no response object: a reply's
            # content comes from the session's settings.
            await self._ws_send({"type": "response.create", "event_id": event.event_id})
            return
        await super().send_client_event(event)

    async def _handle_evt_response_created(self, evt):
        self._pending_response_create = None
        await super()._handle_evt_response_created(evt)

    async def _handle_server_error(self, error: dict[str, Any], event_ids: set[str]):
        """Handle a non-terminal error; the session continues."""
        message = str(error.get("message") or "")
        # An error that doesn't name the client event it answers is matched by
        # what it says.
        if self._pending_response_create and (
            self._pending_response_create in event_ids
            or message.startswith("invalid client event: response.")
            or "already has an active response" in message.lower()
        ):
            await self._handle_response_create_rejected(message)
        if any(expected in message.lower() for expected in _EXPECTED_ERRORS):
            logger.debug(f"{self}: {message}")
            return
        await self.push_error(
            error_msg=f"Zoom Voice Agent error: {message} (tracking_id={self._tracking_id})"
        )

    async def _handle_response_create_rejected(self, message: str):
        """The server refused a response.create; no response.done will follow it."""
        self._pending_response_create = None
        if "already has an active response" in message.lower():
            # Ask again once that response is done, if it's still needed then.
            self._response_create_owed = True
            return
        self._response_active = False
        await self.stop_all_metrics()
        await self.push_frame(LLMFullResponseEndFrame())

    async def _handle_evt_function_call_arguments_done(self, evt):
        # A call without arguments may come with none at all; it still needs a
        # result, or the session waits for one.
        if not evt.arguments.strip():
            evt.arguments = "{}"
        await super()._handle_evt_function_call_arguments_done(evt)

    async def _handle_evt_conversation_item_added(self, evt):
        if evt.item.type == "function_call" and evt.item.call_id and evt.item.id:
            self._call_items[evt.item.call_id] = evt.item.id
        await super()._handle_evt_conversation_item_added(evt)

    async def _send_tool_result(self, tool_call_id: str, result: str | None):
        previous = self._call_items.pop(tool_call_id, None)
        if previous is None:
            await super()._send_tool_result(tool_call_id, result)
            return
        self._open_function_calls.discard(tool_call_id)
        item = openai_events.ConversationItem(
            type="function_call_output", call_id=tool_call_id, output=result
        )
        await self.send_client_event(
            openai_events.ConversationItemCreateEvent(previous_item_id=previous, item=item)
        )

    async def _handle_evt_session_updated(self, evt):
        self._listen_configured = True
        if self._restore_conversation_when_ready:
            self._restore_conversation_when_ready = False
            await self._restore_conversation()
        await super()._handle_evt_session_updated(evt)
        if self._api_session_ready and self._pending_audio:
            audio, self._pending_audio = bytes(self._pending_audio), bytearray()
            await self._append_audio(audio)

    async def _handle_evt_response_done(self, evt):
        if evt.response.status == "completed":
            self._reconnects = 0  # the session works
        await super()._handle_evt_response_done(evt)

    async def _update_settings(self, delta):
        # The OpenAI service's own handling covers OpenAI-only fields; the session
        # update below is built from this service's settings.
        changed = await super(OpenAIRealtimeLLMService, self)._update_settings(delta)
        handled = {"session_properties", "system_instruction", "model", "listen", "speak", "extra"}
        if self._listen_configured:
            if "listen" in changed:
                logger.warning(f"{self}: listen settings apply to the next session, not this one")
            if "model" in changed:
                logger.warning(f"{self}: the model applies to the next session, not this one")
        if changed.keys() & handled:
            await self._send_session_update()
        self._warn_unhandled_updated_settings(changed.keys() - handled)
        return changed

    async def _send_session_update(self):
        properties = assert_given(self._settings.session_properties)
        adapter = self.get_llm_adapter()
        instructions = properties.instructions
        if self._context:
            params = await adapter.get_llm_invocation_params(
                self._context,
                system_instruction=assert_given(self._settings.system_instruction),
                service_tools=properties.tools,
            )
            instructions = params["system_instruction"] or instructions
            tools = cast(list[Any], params["tools"])
        else:
            tools = cast(list[Any], adapter.from_standard_tools(properties.tools))

        update = events.SessionUpdateEvent(
            session=self._build_session(instructions=instructions, tools=tools)
        ).to_wire()
        extra = dict(self._settings.extra or {})
        if self._listen_configured:
            extra.pop("listen", None)  # fixed once the session is configured
        update["session"] = _merge(update["session"], extra)
        await self._ws_send(update)
        self._listen_configured = True

    def _build_session(
        self, *, instructions: str | None, tools: list[dict[str, Any]] | None
    ) -> events.SessionProperties:
        """The Zoom session for the current settings."""
        properties = assert_given(self._settings.session_properties)

        # The model and the listen settings are part of the first configuration
        # only; later updates may not change them.
        first = not self._listen_configured
        think: dict[str, Any] = {}
        if first and self._settings.model:
            think["provider"] = events.Provider(model=self._settings.model)
        if instructions is not None:
            think["instructions"] = instructions
        if tools is not None:
            think["tools"] = tools
        if properties.tool_choice is not None:
            think["tool_choice"] = _dump(properties.tool_choice)
        if properties.max_output_tokens is not None:
            think["response_token_limit"] = properties.max_output_tokens
        if properties.reasoning is not None:
            think["reasoning"] = _dump(properties.reasoning)
        session: dict[str, Any] = {"think": events.Think(**think)}

        if first:
            listen = self._settings.listen
            listen = listen.model_copy() if isinstance(listen, events.Listen) else events.Listen()
            listen.format = events.AudioFormat(rate=self._input_sample_rate)
            session["listen"] = listen

        if not self._is_modality_enabled("audio"):
            if first:
                session["speak"] = None  # text-only replies
            return events.SessionProperties(**session)

        speak = self._settings.speak
        speak = speak.model_copy() if isinstance(speak, events.Speak) else events.Speak()
        output = properties.audio.output if properties.audio else None
        if output and output.voice and "voice" not in speak.model_fields_set:
            speak.voice = output.voice
        if first:
            speak.format = events.AudioFormat(rate=_VOICE_AGENT_OUTPUT_RATE)
        elif "format" in speak.model_fields_set:
            speak = events.Speak(**speak.model_dump(exclude_unset=True, exclude={"format"}))
        if speak.model_fields_set:
            session["speak"] = speak

        return events.SessionProperties(**session)

    async def _send_user_audio(self, frame):
        # The Voice Agent takes audio at the rate the session was configured with,
        # and only once the session is configured; until then the latest audio is held.
        if self._disconnecting:
            return
        audio = frame.audio
        if frame.sample_rate != self._input_sample_rate:
            audio = await self._input_resampler.resample(
                audio, frame.sample_rate, self._input_sample_rate
            )
        if not self._api_session_ready:
            self._pending_audio.extend(audio)
            limit = int(self._input_sample_rate * 2 * _PENDING_AUDIO_SECS)
            del self._pending_audio[: max(0, len(self._pending_audio) - limit)]
            return
        await self._append_audio(audio)

    async def _append_audio(self, audio: bytes):
        for start in range(0, len(audio), _MAX_APPEND_BYTES):
            payload = base64.b64encode(audio[start : start + _MAX_APPEND_BYTES]).decode("utf-8")
            await self.send_client_event(openai_events.InputAudioBufferAppendEvent(audio=payload))
