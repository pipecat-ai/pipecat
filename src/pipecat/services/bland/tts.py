#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bland text-to-speech service implementations.

See https://docs.bland.ai/api-v2/post/tts-ws for the realtime WebSocket API and
https://docs.bland.ai/api-v2/post/tts for the HTTP API.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService, WebsocketTTSService
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

# Rates Bland renders directly; asking for one of these avoids a resample.
_SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)

# Used when the pipeline rate is not one Bland renders. 48 kHz is what BTTS_V3
# generates natively, so it is the shortest path to audio.
_DEFAULT_SAMPLE_RATE = 48000

_DEFAULT_VOICE_ID = "29158307-9893-4149-8a75-bc9ce313d64e"
_READY_TIMEOUT_SECONDS = 10.0
_CLOSE_TIMEOUT_SECONDS = 5.0


@dataclass
class BlandTTSSettings(TTSSettings):
    """Settings for the Bland TTS services.

    Parameters:
        expressiveness: 0.0-1.0. Higher values produce more varied intonation.
        stability: 0.0-1.0. Higher values produce more consistent delivery.
        auto_formatting: Rewrite numbers into the form the voice reads best
            before synthesis. Phone numbers and SSNs, spelled out ("seven three
            two...") or written as bare digits ("7327412065"), are read digit
            by digit in groups. Off unless set.
    """

    expressiveness: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    stability: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    auto_formatting: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


def _default_settings(settings: BlandTTSSettings | None) -> BlandTTSSettings:
    defaults = BlandTTSSettings(
        model=None,
        voice=_DEFAULT_VOICE_ID,
        language=None,
        expressiveness=None,
        stability=None,
        auto_formatting=None,
    )
    if settings is not None:
        defaults.apply_update(settings)
    return defaults


def _resolve_sample_rate(sample_rate: int) -> int:
    """Pick the rate Bland will synthesize at.

    Bland renders a fixed set of rates. A pipeline running at any other rate is
    served at 48 kHz and resampled by the output transport, which costs a
    resample rather than fidelity: the audio frames carry the rate Bland
    actually produced, not the pipeline's.

    Args:
        sample_rate: The pipeline's output rate.

    Returns:
        The rate Bland will synthesize at.
    """
    if sample_rate in _SAMPLE_RATES:
        return sample_rate
    logger.warning(
        f"Bland cannot render {sample_rate} Hz (supports {list(_SAMPLE_RATES)}); "
        f"synthesizing at {_DEFAULT_SAMPLE_RATE} Hz and resampling on output"
    )
    return _DEFAULT_SAMPLE_RATE


def _controls(settings: BlandTTSSettings) -> dict[str, float]:
    controls: dict[str, float] = {}
    expressiveness = assert_given(settings.expressiveness)
    if expressiveness is not None:
        controls["expressiveness"] = expressiveness
    stability = assert_given(settings.stability)
    if stability is not None:
        controls["stability"] = stability
    return controls


def _apply_request_options(request: dict[str, Any], settings: BlandTTSSettings) -> None:
    """Add the settings Bland reads from a session's ``init`` or a ``/v2/tts`` body."""
    if controls := _controls(settings):
        request["controls"] = controls
    auto_formatting = assert_given(settings.auto_formatting)
    if auto_formatting is not None:
        request["auto_formatting"] = auto_formatting


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
    for ``BTTS_V3``.

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
    # The rate Bland synthesizes at, resolved in setup() from the pipeline's.
    _bland_sample_rate: int

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
        if not api_key:
            raise ValueError("Bland API key is required")
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=False,
            pause_frame_processing=True,
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

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the Bland service supports metrics generation.
        """
        return True

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and open the Bland session.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._bland_sample_rate = _resolve_sample_rate(self.sample_rate)
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
        websocket = None
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
            _apply_request_options(init, self._settings)
            await websocket.send(json.dumps(init))

            # `ready` confirms wallet and concurrency admission, so a
            # rejected session fails here rather than on the first turn.
            message = json.loads(
                await asyncio.wait_for(websocket.recv(), timeout=_READY_TIMEOUT_SECONDS)
            )
            if message.get("type") != "ready":
                raise Exception(
                    f"Bland rejected the session: "
                    f"{message.get('code')}: {message.get('message', message)}"
                )
            if (
                message.get("encoding") != "pcm_s16le"
                or message.get("sample_rate") != self._bland_sample_rate
            ):
                raise Exception(
                    "Bland acknowledged an unexpected audio format: "
                    f"{message.get('encoding')} at {message.get('sample_rate')} Hz"
                )

            logger.debug(f"{self}: session ready (session_id: {message.get('session_id')})")
            self._websocket = websocket
            await self._call_event_handler("on_connected")
        except BaseException as e:
            if websocket is not None:
                try:
                    await websocket.close()
                except Exception:
                    pass
            if not isinstance(e, Exception):
                raise
            await self.push_error(error_msg=f"{self} error: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _close_socket(self):
        """Settle and close the socket.

        Split from ``_disconnect_websocket`` so ``run_tts`` can replace a socket
        Bland closed while idle without stopping the metrics of the turn it is
        about to send.
        """
        websocket = self._websocket
        try:
            # Only a live socket can be settled. Bland reaps an idle session after
            # 60s and closes it itself, and teardown runs after that — sending
            # `close` down a corpse raises, and reporting that as a pipeline error
            # turns routine housekeeping into an ErrorFrame the app has to explain.
            if websocket and websocket.state is State.OPEN:
                logger.debug("Disconnecting from Bland")
                # `done` is sent only after the server settles outstanding usage.
                # The receive task has already stopped, so consume it here before
                # starting the WebSocket close handshake.
                await websocket.send(json.dumps({"type": "close"}))
                async with asyncio.timeout(_CLOSE_TIMEOUT_SECONDS):
                    async for raw in websocket:
                        if isinstance(raw, str):
                            message = json.loads(raw)
                            if message.get("type") == "done":
                                break
        except (TimeoutError, ConnectionClosed) as e:
            # A settle that does not complete costs nothing here: the server bills
            # on disconnect regardless. Worth a log, not an error frame.
            logger.debug(f"{self}: close handshake did not complete ({type(e).__name__}: {e})")
        except Exception as e:
            await self.push_error(error_msg=f"{self} error: {e}", exception=e)
        finally:
            if websocket:
                try:
                    await websocket.close()
                except Exception as e:
                    logger.debug(f"{self} failed to close Bland websocket: {e}")
            self._websocket = None

    async def _disconnect_websocket(self):
        await self.stop_all_metrics()
        await self._close_socket()
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

        # `init` fixes the voice, controls and formatting for the life of a
        # session. Nothing else in TTSSettings reaches Bland, so nothing else
        # earns a reconnect.
        if changed.keys() & {"voice", "expressiveness", "stability", "auto_formatting"}:
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

    async def _close_turn(self, context_id: str | None):
        """Stop and close a turn's audio context, if it is still open."""
        if context_id and self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                context_id = self.get_active_audio_context_id()
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
                logger.trace(f"{self}: turn {context_id} started")
            elif msg_type == "utterance_end":
                reason = msg.get("reason")
                if reason == "failed":
                    # The failure's detail is reported from the `error` message
                    # that precedes it.
                    logger.warning(f"{self}: turn {context_id} failed")
                else:
                    logger.trace(f"{self}: turn {context_id} ended as {reason}")
                await self._close_turn(context_id)
            elif msg_type == "error":
                code = msg.get("code")
                if code == "idle_timeout":
                    # Bland reaps a session after 60s without a client message,
                    # which any conversational pause reaches. Reconnecting is
                    # routine housekeeping, not something the app can act on.
                    logger.debug(f"{self}: session reaped after idle timeout")
                else:
                    await self.push_error(
                        error_msg=f"{self} error {code}: {msg.get('message', msg)}"
                    )
                # An error carrying a context_id ends that turn. A refused turn
                # never starts, so no `utterance_end` arrives to close it.
                await self._close_turn(context_id)
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
                # conversational gap reaches easily. The receive task has finished
                # but is still set after a server close, so it has to be cleared or
                # `_connect()` will not restart it.
                if self._receive_task:
                    await self.cancel_task(self._receive_task)
                    self._receive_task = None
                await self._close_socket()
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
    # The rate Bland synthesizes at, resolved in setup() from the pipeline's.
    _bland_sample_rate: int

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
        if not api_key:
            raise ValueError("Bland API key is required")
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=_default_settings(settings),
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session
        self._session_owner = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the Bland service supports metrics generation.
        """
        return True

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service, creating an aiohttp session if one was not provided.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._bland_sample_rate = _resolve_sample_rate(self.sample_rate)
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

        bland_sample_rate = self._bland_sample_rate

        payload: dict[str, Any] = {
            "text": text,
            "voice": self._settings.voice,
            "audio": {
                "encoding": "pcm_s16le",
                "sample_rate": bland_sample_rate,
                # The body is streamed straight into audio frames, so a
                # container's header would be read as the first samples.
                "container": "raw",
            },
        }
        _apply_request_options(payload, self._settings)

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
