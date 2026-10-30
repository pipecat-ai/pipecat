#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ElevenLabs Realtime speech-to-speech service implementation."""

import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from websockets.exceptions import ConnectionClosed

from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    LLMTextFrame,
)
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

ELEVENLABS_REALTIME_DEFAULT_MODEL = "eleven_realtime_v1_mini"

# The server stops waiting for a function call's output after this long.
FUNCTION_CALL_OUTPUT_DEADLINE_SECS = 20.0

_AUDIO_TAG_RE = re.compile(r"\[[^\s][^\]]*\]")
_REPEATED_SPACES_RE = re.compile(r"[ \t]{2,}")

_LIVE_SESSION_KEYS = frozenset({"type", "instructions", "tools", "tool_choice"})


@dataclass
class ElevenLabsRealtimeLLMSettings(OpenAIRealtimeLLMService.Settings):
    """Settings for ElevenLabsRealtimeLLMService."""

    pass


class ElevenLabsRealtimeLLMService(OpenAIRealtimeLLMService):
    """ElevenLabs Realtime speech-to-speech service.

    Connects to the OpenAI-compatible ElevenLabs Realtime API, which speaks
    the OpenAI GA Realtime protocol, and runs the ``eleven_realtime_v1_mini``
    or ``eleven_realtime_v1_max`` model. Voices are ElevenLabs voice IDs or
    preset names (the server defaults to ``"brian"``), and the conversation
    language is set with ``InputAudioTranscription(language=...)``.

    Once the user's audio reaches the server, only the instructions and tools
    can change; other session properties stay as they were when the session
    was configured. Expressive audio tags such as ``[cheerful]`` are removed
    from the bot's transcript, and the API reports no token usage.

    Example::

        llm = ElevenLabsRealtimeLLMService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            settings=ElevenLabsRealtimeLLMService.Settings(
                system_instruction="You are a helpful assistant.",
            ),
        )
    """

    Settings = ElevenLabsRealtimeLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.elevenlabs.io/v1/convai/realtime",
        settings: Settings | None = None,
        max_duration_seconds: int | None = None,
        input_pacing: Literal["realtime", "bursty"] | None = None,
        turn_timeout: float | None = None,
        **kwargs,
    ):
        """Initialize the ElevenLabs Realtime LLM service.

        Args:
            api_key: ElevenLabs API key with the ``convai_write`` scope.
            base_url: WebSocket URL of the Realtime API. Use a residency host
                (e.g. ``wss://api.eu.residency.elevenlabs.io/v1/convai/realtime``)
                to keep data in a region.
            settings: Runtime-updatable settings for this service.
            max_duration_seconds: Longest the session may last, from 60 to 7200
                seconds. The server closes the session when it is reached, after
                600 seconds if unset.
            input_pacing: ``"bursty"`` when input audio arrives faster than real
                time, such as from a file, so turn-taking follows the audio clock.
                The server assumes ``"realtime"`` if unset.
            turn_timeout: Seconds the bot waits for the user to reply before
                re-engaging them, from 1 to 300, or -1 to never re-engage.
            **kwargs: Additional arguments passed to the parent OpenAIRealtimeLLMService.
        """
        default_settings = self.Settings(model=ELEVENLABS_REALTIME_DEFAULT_MODEL)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

        # The model is chosen in session.update; the server ignores a `?model=` query.
        self.base_url = base_url

        if self._is_turn_detection_disabled():
            logger.warning(
                f"{self}: ElevenLabs Realtime can repeat earlier audio in each committed "
                "turn's transcript when turn_detection is disabled."
            )

        self._max_duration_seconds = max_duration_seconds
        self._input_pacing = input_pacing
        self._turn_timeout = turn_timeout
        self._configured_session: dict[str, Any] = {}
        self._audio_tag_tails: dict[str, str] = {}
        self._function_call_opened_at: dict[str, float] = {}

    async def _auth_headers(self) -> dict[str, str]:
        return {"xi-api-key": self.api_key}

    async def send_client_event(self, event: events.ClientEvent):
        """Send a client event to the ElevenLabs Realtime API.

        Args:
            event: The client event to send.
        """
        if not isinstance(event, events.SessionUpdateEvent):
            await super().send_client_event(event)
            return

        message = event.model_dump(exclude_none=True)
        session = message["session"]
        self._add_session_extensions(session)
        if not self._api_session_ready:
            self._configured_session = session
        else:
            message["session"] = self._live_session_update(session)
        await self._ws_send(message)

    def _add_session_extensions(self, session: dict[str, Any]):
        if self._max_duration_seconds is not None:
            session["max_duration_seconds"] = self._max_duration_seconds
        if self._input_pacing is None and self._turn_timeout is None:
            return
        audio_input = session.setdefault("audio", {}).setdefault("input", {})
        if self._input_pacing is not None:
            audio_input["pacing"] = self._input_pacing
        if self._turn_timeout is not None:
            turn_detection = audio_input.get("turn_detection") or {"type": "semantic_vad"}
            turn_detection["turn_timeout"] = self._turn_timeout
            audio_input["turn_detection"] = turn_detection

    def _live_session_update(self, session: dict[str, Any]) -> dict[str, Any]:
        unchangeable = sorted(
            key
            for key in session.keys() - _LIVE_SESSION_KEYS
            if session[key] != self._configured_session.get(key)
        )
        if unchangeable:
            logger.warning(
                f"{self}: ElevenLabs Realtime can't change {', '.join(unchangeable)} "
                "after the conversation has started; keeping the configured values."
            )
        return {key: value for key, value in session.items() if key in _LIVE_SESSION_KEYS}

    async def _send_user_audio(self, frame: InputAudioRawFrame):
        # Audio reaching the server starts the conversation, which fixes most session properties.
        if not self._api_session_ready:
            return
        await super()._send_user_audio(frame)

    async def _send_message_item(self, text: str, role: str = "system"):
        # The server drops system items without an error.
        await super()._send_message_item(text, role="assistant" if role == "assistant" else "user")

    async def _handle_evt_function_call_arguments_done(self, evt):
        self._function_call_opened_at[evt.call_id] = time.monotonic()
        await super()._handle_evt_function_call_arguments_done(evt)

    async def _handle_function_call_result(self, frame: FunctionCallResultFrame):
        # An output the server no longer waits for also swallows the next response.create.
        deadline = time.monotonic() - FUNCTION_CALL_OUTPUT_DEADLINE_SECS
        for call_id in list(self._open_function_calls):
            if self._function_call_opened_at.get(call_id, deadline) <= deadline:
                self._open_function_calls.discard(call_id)
        await super()._handle_function_call_result(frame)

    async def _handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        # The server settles abandoned calls itself; answering one would be a late output.
        self._open_function_calls.discard(frame.tool_call_id)
        self._completed_tool_calls.add(frame.tool_call_id)

    async def _handle_interruption(self):
        # The server abandons open function calls when the user barges in.
        self._open_function_calls.clear()
        self._audio_tag_tails.clear()
        await super()._handle_interruption()

    async def _handle_evt_text_delta(self, evt):
        delta = self._strip_audio_tags(evt.item_id, evt.delta)
        await super()._handle_evt_text_delta(evt.model_copy(update={"delta": delta}))

    async def _handle_evt_audio_transcript_delta(self, evt):
        delta = self._strip_audio_tags(evt.item_id, evt.delta)
        await super()._handle_evt_audio_transcript_delta(evt.model_copy(update={"delta": delta}))

    def _strip_audio_tags(self, item_id: str, delta: str) -> str:
        text = self._audio_tag_tails.pop(item_id, "") + delta
        start = text.rfind("[")
        if start != -1 and "]" not in text[start:]:
            text, self._audio_tag_tails[item_id] = text[:start], text[start:]
        return _REPEATED_SPACES_RE.sub(" ", _AUDIO_TAG_RE.sub("", text))

    async def _handle_evt_response_done(self, evt):
        tails, self._audio_tag_tails = self._audio_tag_tails, {}
        for tail in tails.values():
            if "audio" in self._get_enabled_modalities():
                await self._push_output_transcript_text_frames(tail)
            else:
                await self.push_frame(LLMTextFrame(tail))
        await super()._handle_evt_response_done(evt)

    async def _disconnect(self):
        await super()._disconnect()
        self._audio_tag_tails.clear()
        self._function_call_opened_at.clear()

    async def _receive_task_handler(self):
        websocket = self._websocket
        try:
            await super()._receive_task_handler()
        except ConnectionClosed:
            pass
        if self._disconnecting or websocket is None or websocket.close_code is None:
            return
        await self.push_error(
            error_msg=(
                f"ElevenLabs Realtime closed the connection: "
                f"{websocket.close_code} {websocket.close_reason}"
            )
        )
