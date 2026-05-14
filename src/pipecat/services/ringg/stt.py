#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Ringg speech-to-text service implementation.

This module provides a streaming STT service backed by the official
``ringglabs`` SDK:

- ``RinggSTTService``: SDK-managed WebSocket streaming STT. Forwards
  Pipecat VAD events as ``start_speaking``/``stop_speaking`` so the
  server can endpoint in ``on_final`` mode.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

# Defer SDK/websockets import failures to a clear install hint instead of an
# opaque ModuleNotFoundError the first time anything imports this module.
try:
    import websockets
    from ringglabs.stt import (
        AckEvent,
        AsyncClient,
        ErrorEvent,
        PongEvent,
        ReadyEvent,
        TranscriptEvent,
    )
    from ringglabs.stt.errors import (
        ApiError,
        ProtocolError,
        TransportError,
    )
    from ringglabs.stt.errors import TimeoutError as RinggTimeoutError
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Ringg STT, you need to `pip install pipecat-ai[ringg]`.")
    raise Exception(f"Missing module: {e}")


class RinggSTTParams(BaseModel):
    """Configuration parameters for the Ringg STT service.

    Parameters:
        api_key: API key for authentication with the Ringg service.
        encoding: Audio encoding format (default: ``int16`` for signed 16-bit PCM).
        language: BCP-47-ish language code for transcription (default: ``hi``).
        mode: Processing mode. ``on_final`` causes the server to emit a final
            transcript when the client sends a ``stop_speaking`` event.
        vad_tail_sil_ms: Trailing silence (ms) used by server VAD.
        vad_confidence: Server VAD confidence threshold (0.0-1.0).
        enable_cap_punc: Whether to enable capitalization/punctuation.
        accept_client_vad_events: If True, the server uses client-sent VAD
            events (start/stop speaking) for endpointing instead of (or in
            addition to) its own VAD.
    """

    api_key: str = ""
    encoding: str = "int16"
    language: str = "hi"
    mode: str = "stream"    # use "on_final" to avoid partial transcripts
    vad_tail_sil_ms: int = 200
    vad_confidence: float = 0.55
    enable_cap_punc: bool = True
    accept_client_vad_events: bool = True


class RinggSTTService(STTService):
    """Speech-to-Text service using the ringglabs SDK.

    Delegates WebSocket transport, handshake, and event parsing to the
    official ``ringglabs`` SDK (``AsyncClient`` / ``AsyncStreamSession``).
    Forwards Pipecat VAD frames as ``start_speaking``/``stop_speaking``
    events to the server, which uses them as the endpointing cue in
    ``on_final`` mode.

    Supported features:

    - Streaming interim and final transcripts
    - Server-side capitalization and punctuation
    - Client-driven VAD endpointing via ``VADUserStartedSpeakingFrame`` /
      ``VADUserStoppedSpeakingFrame``
    - Per-utterance processing metrics

    Example::

        stt = RinggSTTService(
            params=RinggSTTParams(api_key="rg-...", language="hi"),
        )
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        sample_rate: int | None = None,
        params: RinggSTTParams | None = None,
        **kwargs,
    ):
        """Initialize the Ringg STT service.

        Args:
            base_url: Optional override for the Ringg API base URL. Passed
                through to the SDK; leave as ``None`` to use the SDK default.
            sample_rate: Sample rate (Hz) of the audio to be streamed. If not
                provided, the value is taken from the ``StartFrame``.
            params: Service configuration. Defaults are used if omitted.
            **kwargs: Additional arguments forwarded to ``STTService``.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        params = params or RinggSTTParams()

        logger.debug(f"Initializing Ringg STT with base_url: {base_url}")

        self._base_url = base_url
        self._params = params
        self._client: AsyncClient | None = None
        self._session = None
        self._receive_task = None
        self._request_id: str | None = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the SDK-backed STT session."""
        await super().start(frame)
        if self._session is not None:
            return

        try:
            self._client = AsyncClient(
                base_url=self._base_url,
                api_key=self._params.api_key or None,
            )
            self._session = self._client.stream(
                sample_rate=self.sample_rate,
                encoding=self._params.encoding,
                language=self._params.language,
                mode=self._params.mode,
                vad_tail_sil_ms=self._params.vad_tail_sil_ms,
                vad_confidence=self._params.vad_confidence,
                enable_cap_punc=self._params.enable_cap_punc,
                accept_client_vad_events=self._params.accept_client_vad_events,
            )

            ready: ReadyEvent = await self._session.open()
            self._request_id = ready.request_id
            logger.debug(f"Ringg STT connected, request_id: {self._request_id}")

            if not self._receive_task:
                self._receive_task = self.create_task(self._receive_task_handler())

            await self._call_event_handler("on_connected")

        except ApiError as e:
            logger.error(f"Ringg STT: API error during connect: {e}")
            await self._call_event_handler("on_connection_error", str(e))
            await self.push_error(f"Ringg STT: {e}", exception=e)
            await self._cleanup()
        except RinggTimeoutError as e:
            logger.error(f"Ringg STT: Connection timeout: {e}")
            await self._call_event_handler("on_connection_error", "Connection timeout")
            await self.push_error(f"Ringg STT: connection timeout: {e}", exception=e)
            await self._cleanup()
        except (TransportError, ProtocolError) as e:
            logger.error(f"Ringg STT: Transport/protocol error: {e}")
            await self._call_event_handler("on_connection_error", str(e))
            await self.push_error(f"Ringg STT: {e}", exception=e)
            await self._cleanup()
        except Exception as e:
            logger.error(f"Ringg STT: Connection error: {e}")
            await self._call_event_handler("on_connection_error", str(e))
            await self.push_error(f"Ringg STT: {e}", exception=e)
            await self._cleanup()

    async def _cleanup(self):
        """Close the SDK session and client."""
        if self._session is not None:
            try:
                await self._session.end("end")
            except Exception:
                pass
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None

        # Cancel rather than await: the events() iterator may block on the
        # websocket until the server closes it, which would hang cleanup.
        if self._receive_task and self._receive_task is not asyncio.current_task():
            await self.cancel_task(self._receive_task)
        self._receive_task = None

        self._request_id = None
        logger.debug("Disconnected from Ringg STT")
        await self._call_event_handler("on_disconnected")

    async def stop(self, frame: EndFrame):
        """Stop the SDK-backed STT session."""
        await super().stop(frame)
        await self._cleanup()

    async def cancel(self, frame: CancelFrame):
        """Cancel the SDK-backed STT session immediately."""
        await super().cancel(frame)
        await self._cleanup()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio bytes to the SDK session."""
        await self.start_processing_metrics()
        if self._session is not None:
            try:
                await self._session.send_audio(audio)
            except TransportError as e:
                logger.warning(f"Ringg STT: Transport closed while sending audio: {e}")
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Ringg STT: WebSocket closed while sending audio")
            except Exception as e:
                logger.error(f"Ringg STT: Error sending audio: {e}")
        await self.stop_processing_metrics()

        yield None

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: str | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Forward Pipecat VAD events to the SDK session.

        Handles:

        - VADUserStartedSpeakingFrame -> ``session.start_speaking()``
        - VADUserStoppedSpeakingFrame -> ``session.stop_speaking()``
        """
        await super().process_frame(frame, direction)

        if self._session is None:
            return

        # Forwarding Pipecat VAD into the SDK is what causes server-side
        # ``on_final`` mode to emit a final transcript: with
        # ``accept_client_vad_events=True`` the server treats these as the
        # endpointing cue instead of relying on its own VAD alone.
        try:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                await self._session.start_speaking()
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                await self._session.stop_speaking()
        except TransportError as e:
            logger.warning(f"Ringg STT: Transport closed while sending VAD event: {e}")
        except Exception as e:
            logger.error(f"Ringg STT: Error sending VAD event: {e}")

    async def _receive_task_handler(self):
        """Consume typed events from the SDK session."""
        if self._session is None:
            return

        try:
            async for event in self._session.events():
                if isinstance(event, TranscriptEvent):
                    transcription = event.transcription or ""
                    logger.debug(
                        f"Ringg STT: Received transcript event: '{transcription}' "
                        f"(final={event.is_final})"
                    )
                    if not transcription.strip():
                        continue

                    if event.is_final:
                        await self.push_frame(
                            TranscriptionFrame(
                                text=transcription,
                                user_id=self._user_id,
                                timestamp=time_now_iso8601(),
                            )
                        )
                        await self._handle_transcription(
                            transcription,
                            is_final=True,
                            language=event.language or self._params.language,
                        )
                        await self.stop_processing_metrics()
                    else:
                        await self.push_frame(
                            InterimTranscriptionFrame(
                                text=transcription,
                                user_id=self._user_id,
                                timestamp=time_now_iso8601(),
                            )
                        )

                elif isinstance(event, (AckEvent, PongEvent)):
                    pass

                elif isinstance(event, ErrorEvent):
                    logger.error(f"Ringg STT: SDK error [{event.code}]: {event.detail}")
                    await self.push_error(
                        f"Ringg STT: {event.detail}",
                        exception=RuntimeError(f"[{event.code}] {event.detail}"),
                    )

                elif isinstance(event, ReadyEvent):
                    logger.debug("Ringg STT: Received additional ready event")

                else:
                    logger.warning(f"Ringg STT: Unknown event type: {type(event).__name__}")

        except RinggTimeoutError as e:
            logger.warning(f"Ringg STT: Receive timeout: {e}")
        except TransportError:
            logger.debug("Ringg STT: Session closed")
        except Exception as e:
            logger.error(f"Ringg STT: Receive error: {e}")
            await self.push_error(f"Ringg STT: {e}", exception=e)
