#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bland text-to-speech service implementations.

See https://docs.bland.ai/api-v2/post/tts-ws for the realtime WebSocket API and
https://docs.bland.ai/api-v2/post/tts for the HTTP API.
"""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import TextAggregationMode, TTSService, WebsocketTTSService
from pipecat.utils.tracing.service_decorators import traced_tts

# Rates Bland renders directly; asking for one of these avoids a resample.
_SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)

# Used when the pipeline rate is not one Bland renders. 48 kHz is what BTTS_V3
# generates natively, so it is the shortest path to audio.
_DEFAULT_SAMPLE_RATE = 48000

_DEFAULT_VOICE_ID = "f04af0e5-1a80-48a9-b02d-52f30d417cfa"


@dataclass
class BlandTTSSettings(TTSSettings):
    """Settings for the Bland TTS services.

    Parameters:
        expressiveness: 0.0-1.0. Higher values produce more varied intonation.
        stability: 0.0-1.0. Higher values produce more consistent delivery.
    """

    expressiveness: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    stability: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


def _default_settings(settings: BlandTTSSettings | None) -> BlandTTSSettings:
    defaults = BlandTTSSettings(
        model=None,
        voice=_DEFAULT_VOICE_ID,
        language=None,
        expressiveness=None,
        stability=None,
    )
    if settings is not None:
        defaults.apply_update(settings)
    return defaults


def _controls(settings: BlandTTSSettings) -> dict[str, float]:
    controls: dict[str, float] = {}
    expressiveness = assert_given(settings.expressiveness)
    if expressiveness is not None:
        controls["expressiveness"] = expressiveness
    stability = assert_given(settings.stability)
    if stability is not None:
        controls["stability"] = stability
    return controls


class BlandTTSService(WebsocketTTSService):
    """Bland realtime WebSocket text-to-speech service.

    Streams speech from Bland's ``/v2/tts/ws`` endpoint over a single connection
    for the whole conversation. LLM tokens are forwarded as they arrive and Bland
    buffers them server-side, choosing its own synthesis boundaries, so no
    sentence tokenizer or character threshold is needed. Pass
    ``text_aggregation_mode=TextAggregationMode.SENTENCE`` to aggregate into
    sentences before sending instead.

    Interruptions send Bland's ``cancel`` message, so barge-in does not tear down
    the connection.

    The voice sets the model; ``expressiveness`` and ``stability`` are calibrated
    for ``BTTS_V3`` and newer.

    Event handlers:

    - on_connected: Called when the websocket connection is established.
    - on_disconnected: Called when the websocket connection is closed.
    - on_connection_error: Called when a websocket connection error occurs.

    Example::

        tts = BlandTTSService(
            api_key=os.getenv("BLAND_API_KEY"),
            settings=BlandTTSService.Settings(
                voice="29158307-9893-4149-8a75-bc9ce313d64e"
            ),
        )
    """

    Settings = BlandTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.bland.ai/v2/tts/ws",
        sample_rate: int | None = None,
        text_aggregation_mode: TextAggregationMode = TextAggregationMode.TOKEN,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Bland WebSocket TTS service.

        Args:
            api_key: Bland API key for authentication.
            url: WebSocket URL for the Bland realtime TTS API. Defaults to
                ``wss://api.bland.ai/v2/tts/ws``.
            sample_rate: Output sample rate in Hz. If None, uses the pipeline
                default. A rate Bland does not render is replaced with 48000 and
                resampled by the output transport.
            text_aggregation_mode: How to aggregate incoming text before sending.
                Defaults to ``TextAggregationMode.TOKEN``, streaming LLM tokens
                straight to Bland for the lowest latency.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to ``WebsocketTTSService``.
        """
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=False,
            text_aggregation_mode=text_aggregation_mode,
            # Bland appends each `speak.text` verbatim, so consecutive sentences
            # would otherwise glue together. Applies in sentence mode only; when
            # streaming tokens the LLM's own whitespace is used as-is.
            append_trailing_space=True,
            settings=_default_settings(settings),
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._receive_task = None
        # Binary frames carry no ID, so audio belongs to the turn Bland announced
        # with `utterance_start`.
        self._utterance_context_id: str | None = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the Bland service supports metrics generation.
        """
        return True

    @property
    def _bland_sample_rate(self) -> int:
        return self.sample_rate if self.sample_rate in _SAMPLE_RATES else _DEFAULT_SAMPLE_RATE

    async def start(self, frame: StartFrame):
        """Start the service and open the Bland session.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
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
        """Open the socket and hold the session at ``ready``."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Bland")

            websocket = await self._websocket_connect(
                self._url, additional_headers={"Authorization": f"Bearer {self._api_key}"}
            )

            init: dict[str, Any] = {
                "type": "init",
                "voice": self._settings.voice,
                "audio": {"encoding": "pcm_s16le", "sample_rate": self._bland_sample_rate},
            }
            if controls := _controls(self._settings):
                init["controls"] = controls
            await websocket.send(json.dumps(init))

            # `ready` also reports the wallet and concurrency admission, so a
            # rejected session fails here rather than on the first turn.
            message = json.loads(await websocket.recv())
            if message.get("type") != "ready":
                raise Exception(
                    f"Bland rejected the session: "
                    f"{message.get('code')}: {message.get('message', message)}"
                )

            logger.debug(f"{self}: session ready (session_id: {message.get('session_id')})")
            self._websocket = websocket
            self._utterance_context_id = None
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error_frame(ErrorFrame(error=f"{self} error: {e}"))
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Bland")
                # `close` cancels any active turn, settles usage, and lets the
                # server reply with `done` before the socket goes away.
                await self._websocket.send(json.dumps({"type": "close"}))
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error_frame(ErrorFrame(error=f"{self} error: {e}"))
        finally:
            await self.remove_active_audio_context()
            self._utterance_context_id = None
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Args:
            delta: A :class:`TTSSettings` (or ``BlandTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        # `init` fixes the voice and controls for the life of a session.
        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the interrupted turn instead of dropping the connection."""
        await self.stop_all_metrics()
        if context_id and self._websocket:
            try:
                await self._websocket.send(json.dumps({"type": "cancel", "context_id": context_id}))
            except Exception as e:
                logger.error(f"{self} error sending cancel message: {e}")
        await super().on_audio_context_interrupted(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """End the active turn so Bland synthesizes its remaining buffered text.

        Args:
            context_id: The turn to end. Falls back to the active context.
        """
        turn_id = context_id or self.get_active_audio_context_id()
        if not turn_id or not self._websocket:
            return
        try:
            await self._websocket.send(json.dumps({"type": "end_of_turn", "context_id": turn_id}))
        except Exception as e:
            logger.error(f"{self} error sending end_of_turn message: {e}")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                context_id = self._utterance_context_id or self.get_active_audio_context_id()
                await self.stop_ttfb_metrics()
                await self.append_to_audio_context(
                    context_id,
                    TTSAudioRawFrame(message, self._bland_sample_rate, 1, context_id=context_id),
                )
                continue

            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON message: {message}")
                continue

            msg_type = msg.get("type")
            context_id = msg.get("context_id")

            if msg_type == "utterance_start":
                self._utterance_context_id = context_id
            elif msg_type == "utterance_end":
                self._utterance_context_id = None
                reason = msg.get("reason")
                if reason == "complete":
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                    await self.remove_audio_context(context_id)
                elif reason == "failed":
                    await self.push_error(error_msg=f"{self} turn {context_id} failed")
                    await self.remove_audio_context(context_id)
                else:
                    # `preempted` and `cancelled` follow an interruption, which
                    # already tore the context down.
                    logger.trace(f"{self}: turn {context_id} ended as {reason}")
            elif msg_type == "error":
                await self.push_error(
                    error_msg=f"{self} error {msg.get('code')}: {msg.get('message', msg)}"
                )
            elif msg_type == "done":
                logger.debug(f"{self}: session settled (session_id: {msg.get('session_id')})")
            else:
                logger.debug(f"Received unknown message type: {msg}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Append a text delta to the current turn.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Nothing directly; audio arrives on the receive task.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                # Bland ends a session after 60s without a client message, which a
                # conversational gap reaches easily. Cycle rather than reconnect:
                # after a server close the receive task has finished but is still
                # set, so a plain _connect() would not restart it.
                await self._disconnect()
                await self._connect()

            await self._get_websocket().send(
                json.dumps({"type": "speak", "context_id": context_id, "text": text})
            )

            await self.start_tts_usage_metrics(text)

            # The audio frames will be handled in _receive_messages
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


class BlandHttpTTSService(TTSService):
    """Bland HTTP text-to-speech service.

    Generates speech with Bland's ``/v2/tts`` endpoint, which takes the complete
    text in one request. Voice agents should prefer :class:`BlandTTSService`,
    which streams text into a realtime session.
    """

    Settings = BlandTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.bland.ai/v2",
        sample_rate: int | None = None,
        aiohttp_session: aiohttp.ClientSession | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Bland HTTP TTS service.

        Args:
            api_key: Bland API key for authentication.
            base_url: Base URL for the Bland API. Defaults to
                ``https://api.bland.ai/v2``.
            sample_rate: Output sample rate in Hz. If None, uses the pipeline
                default.
            aiohttp_session: Optional shared aiohttp session. When omitted, the
                service creates and owns one.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to ``TTSService``.
        """
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=_default_settings(settings),
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session
        self._session_owner = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the Bland service supports metrics generation.
        """
        return True

    async def start(self, frame):
        """Start the service, creating an aiohttp session if one was not provided."""
        await super().start(frame)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

    async def stop(self, frame):
        """Stop the service and release an owned aiohttp session."""
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame):
        """Cancel the service and release an owned aiohttp session."""
        await super().cancel(frame)
        await self._close_session()

    async def cleanup(self):
        """Release Bland TTS resources at teardown."""
        await super().cleanup()
        await self._close_session()

    async def _close_session(self):
        if self._session_owner and self._session and not self._session.closed:
            await self._session.close()
        if self._session_owner:
            self._session = None

    def _bland_sample_rate(self) -> int:
        return self.sample_rate if self.sample_rate in _SAMPLE_RATES else _DEFAULT_SAMPLE_RATE

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Bland's ``/v2/tts`` endpoint.

        Args:
            text: The text to synthesize.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

        bland_sample_rate = self._bland_sample_rate()

        payload: dict[str, Any] = {
            "text": text,
            "voice": self._settings.voice,
            "audio": {"encoding": "pcm_s16le", "sample_rate": bland_sample_rate},
        }
        if controls := _controls(self._settings):
            payload["controls"] = controls

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with self._session.post(
                f"{self._base_url}/tts", json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    yield ErrorFrame(error=await _error_message(response))
                    return

                await self.start_tts_usage_metrics(text)

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    in_sample_rate=bland_sample_rate,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()


async def _error_message(response: aiohttp.ClientResponse) -> str:
    """Unwrap the v2 ``{"error": {"code", "message"}}`` envelope, falling back to the raw body."""
    try:
        payload = await response.json()
    except Exception:
        body = await response.text(errors="ignore")
        return f"Error getting audio (status: {response.status}, error: {body})"

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        detail = ": ".join(str(v) for v in (error.get("code"), error.get("message")) if v)
        if detail:
            return f"Error getting audio (status: {response.status}, error: {detail})"
    return f"Error getting audio (status: {response.status}, error: {payload})"
