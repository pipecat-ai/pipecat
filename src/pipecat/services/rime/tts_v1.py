#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime WebSocket v1 text-to-speech service implementation."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import ClassVar, Literal
from urllib.parse import urlsplit

from google.protobuf import json_format
from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

try:
    from rime_api import text_to_speech_pb2 as proto
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use RimeV1TTSService, you need to `uv add "pipecat-ai[rime]"`.')
    raise ImportError(f"Missing module: {e}") from e

WebSocketProtocol = Literal["binary", "json"]

_MODEL_WEBSOCKET_URLS = {
    "coda": "wss://coda.api.rime.ai/ws",
    "mistv3": "wss://mist.api.rime.ai/ws",
}
_PROTOCOL_VERSION = 1
_READY_TIMEOUT_S = 10.0
_SUBPROTOCOLS: dict[str, str] = {"binary": "rime.v1.binary", "json": "rime.v1.json"}
_ERROR_CATEGORIES: dict[str, ErrorCategory] = {
    "invalid_input": ErrorCategory.INVALID_REQUEST,
    "unauthenticated": ErrorCategory.AUTHENTICATION,
    "permission_denied": ErrorCategory.AUTHORIZATION,
    "not_found": ErrorCategory.INVALID_REQUEST,
    "resource_exhausted": ErrorCategory.RATE_LIMIT,
    "timeout": ErrorCategory.CONNECTIVITY,
    "unavailable": ErrorCategory.CONNECTIVITY,
    "unimplemented": ErrorCategory.SERVER,
    "internal": ErrorCategory.SERVER,
}


def language_to_rime_v1_language(language: Language) -> str:
    """Convert a pipecat Language to the BCP-47 tag Rime WebSocket v1 expects.

    Args:
        language: The pipecat Language enum value.

    Returns:
        The base BCP-47 tag (e.g., 'en' for English). A region-qualified
        language falls back to its base tag.
    """
    LANGUAGE_MAP = {
        Language.AR: "ar",
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.IT: "it",
        Language.JA: "ja",
        Language.PT: "pt",
    }
    return resolve_language(language, LANGUAGE_MAP)


def _error_message(error: proto.WebSocketError) -> str:
    message = f"Rime error ({error.kind}): {error.message}"
    if error.HasField("request_id"):
        message += f" (request ID: {error.request_id})"
    return message


class _RimeV1Error(Exception):
    """Connection-level error reported by Rime."""

    def __init__(self, error: proto.WebSocketError):
        super().__init__(_error_message(error))
        self.kind = error.kind


@dataclass
class RimeV1TTSSettings(TTSSettings):
    """Settings for RimeV1TTSService.

    Each context uses the settings in effect when it starts.

    Parameters:
        model: Required at construction. ``"coda"`` and ``"mistv3"`` select their
            public endpoints unless ``websocket_url`` is provided.
        time_scale_factor: Audio playback speed factor. Values above 1.0 slow down
            the audio; values below 1.0 speed it up. Not supported by mistv2.
        text_lookahead_tokens: Number of input tokens to collect before audio
            starts (Coda only).
        pause_between_brackets: Whether to add pauses between bracketed content (Mist only).
        phonemize_between_brackets: Whether to phonemize bracketed content (Mist only).
        save_oovs: Whether to save out-of-vocabulary words (Mist only).
    """

    time_scale_factor: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_lookahead_tokens: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pause_between_brackets: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    phonemize_between_brackets: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    save_oovs: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[dict[str, str]] = {"speaker": "voice"}


class RimeV1TTSService(WebsocketTTSService):
    """Text-to-Speech service using Rime's WebSocket v1 API.

    Sends protobuf envelopes, in binary or proto3 JSON encoding, over one
    connection that carries a synthesis context per Pipecat turn. Interruptions
    cancel the turn's context without reconnecting. Text is sent as whole
    sentences by default; with ``TextAggregationMode.TOKEN``, LLM tokens are
    sent as they arrive and Rime accumulates them into sentences. ``settings.model``
    is required and selects the public endpoint at construction: ``"coda"`` uses
    ``wss://coda.api.rime.ai/ws`` and ``"mistv3"`` uses ``wss://mist.api.rime.ai/ws``.
    Pass ``websocket_url`` to use a custom deployment. The model is not inferred
    from the URL hostname or path. A context leaves out, with a warning, any
    setting its model does not accept. WebSocket v1 does not report word timestamps.

    Requires the ``rime`` extra: ``uv add "pipecat-ai[rime]"``.
    """

    Settings = RimeV1TTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        websocket_url: str | None = None,
        websocket_protocol: WebSocketProtocol = "binary",
        sample_rate: int | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Rime WebSocket v1 TTS service.

        Args:
            api_key: Rime API key for authentication.
            websocket_url: Rime WebSocket v1 endpoint override. Defaults to the
                public endpoint for ``settings.model``. Custom URLs can use any path.
                Models without a public default require an explicit URL.
            websocket_protocol: Envelope encoding, ``"binary"`` or ``"json"``.
            sample_rate: Audio sample rate in Hz.
            text_aggregation_mode: How to aggregate text before synthesis.
                Defaults to ``TextAggregationMode.SENTENCE``.
                ``TextAggregationMode.TOKEN`` streams LLM tokens and has Rime
                accumulate them into sentences.
            settings: Settings with an explicit ``model``. Other fields can be
                updated at runtime; changing ``model`` does not switch endpoints.
            **kwargs: Additional arguments passed to parent class.
        """
        if websocket_protocol not in _SUBPROTOCOLS:
            raise ValueError('websocket_protocol must be "binary" or "json"')

        default_settings = self.Settings(
            model=None,
            voice=None,
            language=None,
            time_scale_factor=None,
            text_lookahead_tokens=None,
            pause_between_brackets=None,
            phonemize_between_brackets=None,
            save_oovs=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        model = assert_given(default_settings.model)
        if not model:
            raise ValueError("Rime WebSocket v1 requires settings.model")
        if websocket_url is None:
            websocket_url = _MODEL_WEBSOCKET_URLS.get(model)
            if websocket_url is None:
                raise ValueError("websocket_url is required for models other than coda and mistv3")
        parts = urlsplit(websocket_url)
        if parts.scheme not in ("ws", "wss") or parts.hostname is None:
            raise ValueError("Rime WebSocket v1 websocket_url must be an absolute ws or wss URL")

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            push_text_frames=False,
            push_start_frame=True,
            pause_frame_processing=True,
            append_trailing_space=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        # Keep spelled-out text in one aggregate so a split never breaks a tag.
        self._text_aggregator = SkipTagsAggregator(
            [("spell(", ")")], aggregation_type=self._text_aggregation_mode
        )

        self._api_key = api_key
        self._websocket_url = websocket_url
        self._websocket_protocol = websocket_protocol
        self._endpoint_model = model

        self._receive_task = None
        # Contexts open on the current connection, from their start until done,
        # error or cancel. Audio for any other context is stale and dropped.
        self._contexts: set[str] = set()
        # Contexts Rime closed with an error; the rest of their turn is skipped.
        self._failed_contexts: set[str] = set()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Rime service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert pipecat language to the BCP-47 tag Rime WebSocket v1 expects.

        Args:
            language: The language to convert.

        Returns:
            The BCP-47 language tag.
        """
        return language_to_rime_v1_language(language)

    # A set of Rime-specific helpers for text transformations
    @staticmethod
    def SPELL(text: str) -> str:
        """Wrap text in Rime spell function."""
        return f"spell({text})"

    @staticmethod
    def PAUSE_TAG(seconds: float) -> str:
        """Convenience method to create a pause tag."""
        return f"<{seconds * 1000}>"

    def _warn_ignored(self, name: str, model: str):
        logger.warning(f"{self}: {name} is not applied, {model} does not accept it")

    def _build_start_request(self, context_id: str) -> proto.WebSocketRequest:
        """Build the start envelope that fixes a context's settings.

        Settings the endpoint's model does not accept are left out with a warning.
        """
        s = self._settings
        model = self._endpoint_model
        if assert_given(s.model) != model:
            logger.warning(f"{self}: model {s.model} is not applied, the endpoint serves {model}")

        audio_parameters = proto.AudioParameters(
            audio_format="audio/pcm", sampling_rate=self.sample_rate
        )
        time_scale_factor = assert_given(s.time_scale_factor)
        if time_scale_factor is not None:
            if model == "mistv2":
                self._warn_ignored("time_scale_factor", model)
            else:
                audio_parameters.time_scale_factor = time_scale_factor

        start = proto.SynthesisRequest(text="", audio_parameters=audio_parameters)
        if self._is_streaming_tokens:
            start.split_strategy = proto.SPLIT_STRATEGY_ACCUMULATE_SENTENCE
        voice = assert_given(s.voice)
        if voice is not None:
            start.speaker = voice
        # The stored language is always a BCP-47 tag; the field type also allows Language.
        language = assert_given(s.language)
        if isinstance(language, str):
            start.language = language

        text_lookahead_tokens = assert_given(s.text_lookahead_tokens)
        if text_lookahead_tokens is not None:
            if model == "coda":
                start.coda_parameters.text_lookahead_tokens = text_lookahead_tokens
            else:
                self._warn_ignored("text_lookahead_tokens", model)

        mist_settings = {
            "pause_between_brackets": assert_given(s.pause_between_brackets),
            "phonemize_between_brackets": assert_given(s.phonemize_between_brackets),
            "save_oovs": assert_given(s.save_oovs),
        }
        for name, value in mist_settings.items():
            if value is None:
                continue
            if model.startswith("mist"):
                setattr(start.mist_parameters, name, value)
            else:
                self._warn_ignored(name, model)
        return proto.WebSocketRequest(context_id=context_id, start=start)

    def _encode(self, request: proto.WebSocketRequest) -> str | bytes:
        if self._websocket_protocol == "binary":
            return request.SerializeToString()
        return json_format.MessageToJson(request, indent=None)

    def _decode(self, message: str | bytes) -> proto.WebSocketResponse:
        response = proto.WebSocketResponse()
        if self._websocket_protocol == "binary":
            if not isinstance(message, bytes):
                raise ValueError("Rime v1 sent a text frame on the binary protocol")
            response.ParseFromString(message)
        else:
            json_format.Parse(message, response, ignore_unknown_fields=True)
        return response

    async def _send(self, request: proto.WebSocketRequest):
        await self._get_websocket().send(self._encode(request))

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        websocket = None
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Rime WebSocket v1")
            subprotocol = _SUBPROTOCOLS[self._websocket_protocol]
            websocket = await self._websocket_connect(
                self._websocket_url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                subprotocols=[subprotocol],
            )
            if websocket.subprotocol != subprotocol:
                raise ConnectionError(f"Rime v1 did not accept the {subprotocol} subprotocol")

            # Rime sends `ready` once its engine can accept contexts.
            self._websocket = websocket
            ready = self._decode(await asyncio.wait_for(websocket.recv(), _READY_TIMEOUT_S))
            if ready.WhichOneof("payload") == "error":
                raise _RimeV1Error(ready.error)
            if ready.WhichOneof("payload") != "ready" or ready.ready.protocol != _PROTOCOL_VERSION:
                raise ConnectionError("Rime v1 did not send a supported ready event")

            self._contexts.clear()
            self._failed_contexts.clear()
            await self._call_event_handler("on_connected")
        except Exception as e:
            category = _ERROR_CATEGORIES.get(e.kind) if isinstance(e, _RimeV1Error) else None
            await self.push_error(
                error_msg=f"Error connecting: {e}", exception=e, category=category
            )
            if websocket:
                await websocket.close()
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug(f"{self}: Disconnecting from Rime WebSocket v1")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)
        finally:
            self._websocket = None
            self._contexts.clear()
            self._failed_contexts.clear()
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the Rime context when the bot is interrupted."""
        await self.stop_all_metrics()
        self._failed_contexts.discard(context_id)
        if context_id in self._contexts:
            self._contexts.discard(context_id)
            await self._send(proto.WebSocketRequest(context_id=context_id, cancel={}))
        await super().on_audio_context_interrupted(context_id)

    async def on_turn_context_completed(self):
        """Close a failed context's audio once its turn has no more text."""
        context_id = self._turn_context_id
        await super().on_turn_context_completed()
        if context_id in self._failed_contexts:
            await self._close_failed_context(context_id)

    async def _close_failed_context(self, context_id: str):
        self._failed_contexts.discard(context_id)
        if self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """End the context's input so Rime synthesizes the rest of its text.

        Args:
            context_id: The specific context to flush. If None, falls back to the
                currently active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket or flush_id not in self._contexts:
            return
        logger.trace(f"{self}: flushing audio")
        await self._send(proto.WebSocketRequest(context_id=flush_id, end={}))

    async def _receive_messages(self):
        async for message in self._get_websocket():
            response = self._decode(message)
            payload = response.WhichOneof("payload")
            ctx_id = response.context_id

            if payload == "error" and not ctx_id:
                await self.push_error(
                    error_msg=_error_message(response.error),
                    category=_ERROR_CATEGORIES.get(response.error.kind),
                )
                continue
            if ctx_id not in self._contexts or not self.audio_context_available(ctx_id):
                continue

            if payload == "audio":
                if response.audio:
                    frame = TTSAudioRawFrame(
                        audio=response.audio,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=ctx_id,
                    )
                    await self.append_to_audio_context(ctx_id, frame)
            elif payload == "done":
                self._contexts.discard(ctx_id)
                await self.stop_ttfb_metrics()
                await self.append_to_audio_context(ctx_id, TTSStoppedFrame(context_id=ctx_id))
                await self.remove_audio_context(ctx_id)
            elif payload == "error":
                self._contexts.discard(ctx_id)
                self._failed_contexts.add(ctx_id)
                await self.stop_all_metrics()
                # One failed context leaves the service usable for the next turn.
                category = _ERROR_CATEGORIES.get(response.error.kind, ErrorCategory.UNKNOWN)
                await self.push_error(
                    error_msg=_error_message(response.error),
                    category=ErrorCategory.UNKNOWN if category.is_permanent else category,
                )
                # While the turn is still sending text, its later sentences reuse
                # this audio context; closing it now would make the base reopen an
                # empty one for them. on_turn_context_completed closes it instead.
                if ctx_id != self._turn_context_id:
                    await self._close_failed_context(ctx_id)

    async def _text_messages(self, text: str) -> list[str]:
        """Split prepared text into the text messages Rime v1 expects.

        Rime accumulates streamed tokens itself, so each token is one message.
        Otherwise each message is one sentence.
        """
        if self._is_streaming_tokens:
            return [text]
        aggregator = SkipTagsAggregator(
            [("spell(", ")")], aggregation_type=TextAggregationMode.SENTENCE
        )
        sentences = [
            self._prepare_text_for_tts(aggregate.text)
            async for aggregate in aggregator.aggregate(text)
        ]
        remaining = await aggregator.flush()
        if remaining and remaining.text:
            sentences.append(self._prepare_text_for_tts(remaining.text))
        return sentences

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Rime's WebSocket v1 API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        if context_id in self._failed_contexts:
            return
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if context_id not in self._contexts:
                    await self._send(self._build_start_request(context_id))
                    self._contexts.add(context_id)
                for sentence in await self._text_messages(text):
                    await self._send(proto.WebSocketRequest(context_id=context_id, text=sentence))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
