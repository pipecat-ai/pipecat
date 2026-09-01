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

_DEFAULT_VOICE_ID = "2f29fdbb-c55e-4add-9c7c-93437ebf379d"
_READY_TIMEOUT_SECONDS = 10.0
_CLOSE_TIMEOUT_SECONDS = 5.0


@dataclass
class BlandTTSSettings(TTSSettings):
    """Settings for the Bland TTS services.

    Parameters:
        expressiveness: 0.0-1.0. Higher values produce more varied intonation.
        stability: 0.0-1.0. Higher values produce more consistent delivery.
    """

    expressiveness: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    stability: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


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
                voice="2f29fdbb-c55e-4add-9c7c-93437ebf379d"
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
        # The turn whose deltas have reached the current socket, and the turn that
        # can no longer be completed. One slot each: the protocol carries one turn
        # at a time, so a new context supersedes.
        self._sent_context_id: str | None = None
        self._abandoned_context_id: str | None = None

    def _abandon_turn(self, context_id: str) -> None:
        """Stop feeding a turn that cannot finish, without ending the session."""
        self._abandoned_context_id = context_id
        if self._utterance_context_id == context_id:
            self._utterance_context_id = None
        # No longer in flight, so a socket closing later must not report it a
        # second time as a turn it lost.
        if self._sent_context_id == context_id:
            self._sent_context_id = None

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
            if controls := _controls(self._settings):
                init["controls"] = controls
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
            self._utterance_context_id = None
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
        """Settle and close the socket, leaving pipeline state untouched.

        Split from ``_disconnect_websocket`` so a mid-turn reconnect can replace
        the transport without destroying the audio context of the turn it is about
        to resume.
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
            self._utterance_context_id = None
            self._sent_context_id = None
            self._websocket = None

    async def _disconnect_websocket(self):
        await self.stop_all_metrics()
        await self._close_socket()
        await self.remove_active_audio_context()
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

        # `init` fixes the voice and controls for the life of a session. Nothing
        # else in TTSSettings reaches Bland, so nothing else earns a reconnect.
        if changed.keys() & {"voice", "expressiveness", "stability"}:
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
        # A cancelled turn is over locally straight away, without waiting for the
        # server's `utterance_end`: a socket dying before that arrives would
        # otherwise report the turn as one the connection lost mid-flight.
        if context_id:
            self._abandon_turn(context_id)
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
        try:
            await self._read_until_closed()
        finally:
            # The loop only exits when the socket is gone. A turn still in flight
            # dies with it: turn state lives in the session, so the reconnect the
            # base class is about to perform knows nothing about it. Feeding the
            # rest of the turn into the new session would speak the tail of a
            # sentence as if it were the whole thing.
            lost = self._sent_context_id
            if lost is not None:
                self._abandon_turn(lost)
                await self.push_error(
                    error_msg=f"{self} lost the connection mid-turn; turn {lost} was dropped"
                )
                if self.audio_context_available(lost):
                    await self.append_to_audio_context(lost, TTSStoppedFrame(context_id=lost))
                    await self.remove_audio_context(lost)

    async def _read_until_closed(self):
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
                # Terminated, so it is no longer a turn a dying socket could lose.
                if self._sent_context_id == context_id:
                    self._sent_context_id = None
                reason = msg.get("reason")
                if reason == "complete":
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                    await self.remove_audio_context(context_id)
                elif reason == "failed":
                    # The server sends the detail as an `error` frame just before
                    # this terminal, and that branch abandons the turn — so an
                    # already-abandoned context has been reported and does not need
                    # a second, vaguer frame. Report only if nothing did.
                    if self._abandoned_context_id != context_id:
                        await self.push_error(error_msg=f"{self} turn {context_id} failed")
                    self._abandon_turn(context_id)
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                    await self.remove_audio_context(context_id)
                else:
                    logger.trace(f"{self}: turn {context_id} ended as {reason}")
                    # Preempted or cancelled: over for good either way. Deltas
                    # still arriving under that context_id have the server admit
                    # and bill a fresh turn, speaking a sentence tail nobody
                    # asked for.
                    self._abandon_turn(context_id)
                    # An explicit Pipecat interruption normally removed this
                    # context already, but a server-side preemption can arrive
                    # first, so the guard closes whichever side still owns it.
                    if context_id and self.audio_context_available(context_id):
                        await self.append_to_audio_context(
                            context_id, TTSStoppedFrame(context_id=context_id)
                        )
                        await self.remove_audio_context(context_id)
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
                # Every error carrying a context_id ends that turn, in one of two
                # shapes. An admission refusal — turn admission happens on the
                # first `speak` — never creates the turn, so no `utterance_end`
                # arrives to release Pipecat's pre-created audio context; that is
                # released here. A mid-turn rejection such as `context_overflow`
                # is followed by `utterance_end(failed)`. Abandoning covers both:
                # either way the remaining deltas must stop.
                if context_id:
                    self._abandon_turn(context_id)
                    if self.audio_context_available(context_id):
                        await self.append_to_audio_context(
                            context_id, TTSStoppedFrame(context_id=context_id)
                        )
                        await self.remove_audio_context(context_id)
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
        if context_id == self._abandoned_context_id:
            # This turn can no longer be completed and has already been reported.
            # Its remaining deltas would only ask again, once per token.
            yield None
            return

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                # Bland ends a session after 60s without a client message, which a
                # conversational gap reaches easily. Cycle the socket rather than
                # calling `_disconnect()`: that removes the active audio context —
                # the context of the very turn this call is about to send. The
                # receive task has finished but is still set after a server close,
                # so it has to be cleared or `_connect()` will not restart it.
                #
                # A turn the socket died under is abandoned by the receive loop
                # rather than here: the base class reconnects the moment that loop
                # exits, so by the time the next delta arrives the socket is healthy
                # again and this branch cannot see the failure.
                if self._receive_task:
                    await self.cancel_task(self._receive_task)
                    self._receive_task = None
                await self._close_socket()
                await self._connect()

            await self._get_websocket().send(
                json.dumps({"type": "speak", "context_id": context_id, "text": text})
            )
            self._sent_context_id = context_id

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
