#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Magpie TTS service backed by an AWS SageMaker endpoint."""

import asyncio
import base64
import json
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import aioboto3
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.aws.sagemaker.bidi_client import SageMakerBidiClient
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService, TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class NvidiaSageMakerTTSSettings(TTSSettings):
    """Settings for NVIDIA SageMaker TTS services.

    Parameters:
        voice: NIM voice name (e.g. ``Magpie-Multilingual.EN-US.Aria``).
        language: BCP-47 language code passed to NIM (e.g. ``en-US``).
    """


class NvidiaSageMakerHTTPTTSService(TTSService):
    """NVIDIA Magpie TTS service that calls a SageMaker HTTP endpoint.

    Sends each text segment to the wrapper's ``POST /invocations`` endpoint
    as a JSON body and streams the raw PCM audio response back to bot
    as :class:`TTSAudioRawFrame` frames.

    Example::

        tts = NvidiaSageMakerHTTPTTSService(
            endpoint_name=os.getenv("SAGEMAKER_MAGPIE_ENDPOINT_NAME"),
            region=os.getenv("AWS_REGION", "us-west-2"),
            settings=NvidiaSageMakerHTTPTTSService.Settings(
                voice="Magpie-Multilingual.EN-US.Aria",
                language="en-US",
            ),
        )
    """

    Settings = NvidiaSageMakerTTSSettings

    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str = "us-west-2",
        sample_rate: int | None = None,
        settings: NvidiaSageMakerTTSSettings | None = None,
        **kwargs,
    ):
        """Initialize the SageMaker HTTP TTS service.

        Args:
            endpoint_name: Name of the deployed SageMaker endpoint.
            region: AWS region where the endpoint lives.
            sample_rate: Output sample rate in Hz. Defaults to bot's pipeline rate.
            settings: Runtime-updatable settings (voice, language).
            **kwargs: Forwarded to :class:`TTSService`.
        """
        default_settings = self.Settings(
            model="magpie",
            voice="Magpie-Multilingual.EN-US.Aria",
            language="en-US",
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._endpoint_name = endpoint_name
        self._region = region
        self._client = None
        self._client_ctx = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics generation.
        """
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, frame: StartFrame):
        """Start the TTS service and create the SageMaker client.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        session = aioboto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=self._region,
        )
        self._client_ctx = session.client("sagemaker-runtime")
        self._client = await self._client_ctx.__aenter__()
        logger.debug(f"{self}: connected to SageMaker endpoint '{self._endpoint_name}'")

    async def _close_client(self):
        if self._client_ctx is not None:
            try:
                await self._client_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"{self}: error closing SageMaker client: {e}")
            self._client_ctx = None
            self._client = None

    async def stop(self, frame: EndFrame):
        """Stop the TTS service and close the SageMaker client.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._close_client()

    async def cancel(self, frame: CancelFrame):
        """Cancel the TTS service and close the SageMaker client.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._close_client()

    # ── Synthesis ─────────────────────────────────────────────────────────────

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize text via SageMaker and yield a single PCM audio frame.

        Args:
            text: The text to synthesize.
            context_id: Pipecat audio context identifier.

        Yields:
            :class:`TTSAudioRawFrame` chunks of signed 16-bit mono PCM.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        text = text.strip()
        if not text or not any(c.isalnum() for c in text):
            return

        try:
            assert self._client is not None
            body = json.dumps(
                {
                    "text": text,
                    "voice_name": self._settings.voice,
                    "language_code": self._settings.language,
                    "sample_rate_hz": self.sample_rate,
                }
            )

            response = await self._client.invoke_endpoint(
                EndpointName=self._endpoint_name,
                ContentType="application/json",
                Accept="application/octet-stream",
                Body=body,
            )

            if "Body" not in response:
                yield ErrorFrame(error="SageMaker TTS returned no audio stream")
                return

            first_chunk = True
            async for chunk in response["Body"].iter_chunks(chunk_size=self.chunk_size):
                if chunk:
                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        first_chunk = False
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
        except Exception as e:
            logger.error(f"{self}: SageMaker TTS error: {e}")
            yield ErrorFrame(error=f"SageMaker TTS error: {e}")

        await self.start_tts_usage_metrics(text)


class NvidiaSageMakerTTSService(InterruptibleTTSService):
    """NVIDIA Magpie TTS service using SageMaker bidirectional streaming.

    Maintains a persistent HTTP/2 bidi-stream session to the SageMaker endpoint
    for the lifetime of the pipeline.  Each text segment is sent as NIM realtime
    events; audio chunks arrive asynchronously and are pushed as
    :class:`TTSAudioRawFrame` frames.

    Example::

        tts = NvidiaSageMakerTTSService(
            endpoint_name=os.getenv("SAGEMAKER_MAGPIE_ENDPOINT_NAME"),
            region=os.getenv("AWS_REGION", "us-west-2"),
            settings=NvidiaSageMakerTTSService.Settings(
                voice="Magpie-Multilingual.EN-US.Aria",
                language="en-US",
            ),
        )
    """

    Settings = NvidiaSageMakerTTSSettings

    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str = "us-west-2",
        sample_rate: int | None = None,
        settings: NvidiaSageMakerTTSSettings | None = None,
        **kwargs,
    ):
        """Initialize the SageMaker WebSocket TTS service.

        Args:
            endpoint_name: Name of the deployed SageMaker endpoint.
            region: AWS region where the endpoint lives.
            sample_rate: Output sample rate in Hz. Defaults to pipeline rate.
            settings: Runtime-updatable settings (voice, language).
            **kwargs: Forwarded to :class:`InterruptibleTTSService`.
        """
        default_settings = self.Settings(
            model="magpie",
            voice="Magpie-Multilingual.EN-US.Aria",
            language="en-US",
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            pause_frame_processing=True,
            append_trailing_space=True,
            settings=default_settings,
            **kwargs,
        )

        self._endpoint_name = endpoint_name
        self._region = region
        self._client: SageMakerBidiClient | None = None
        self._receive_task = None
        self._speech_completed_event = asyncio.Event()
        self._audio_buffer = b""
        self._playback_started = False

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics generation.
        """
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, frame: StartFrame):
        """Start the TTS service and connect to the SageMaker endpoint.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the TTS service and disconnect from the SageMaker endpoint.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the TTS service and disconnect from the SageMaker endpoint.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    # ── Connection management (WebsocketService abstract interface) ────────────

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._client and self._client.is_active and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        if self._client and self._client.is_active:
            return

        logger.debug(
            f"{self}: connecting to SageMaker bidi-stream endpoint '{self._endpoint_name}'"
        )
        try:
            self._client = SageMakerBidiClient(
                endpoint_name=self._endpoint_name,
                region=self._region,
                model_query_string=None,
                model_invocation_path=None,
            )
            await self._client.start_session()
            await self._send_session_config()
            logger.debug(f"{self}: connected")
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self}: connection error: {e}")
            self._client = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            if self._client and self._client.is_active:
                logger.debug(f"{self}: disconnecting")
                try:
                    await self._client.send_json({"type": "session.end"})
                except Exception as e:
                    logger.warning(f"{self}: error sending session.end: {e}")
                await self._client.close_session()
                logger.debug(f"{self}: disconnected")
        except Exception as e:
            logger.warning(f"{self}: error during disconnect: {e}")
        finally:
            self._client = None
            await self._call_event_handler("on_disconnected")

    async def _verify_connection(self):
        active = self._client and self._client.is_active
        logger.info(f"{self}: verifying if websocket connection is active {active}")
        return active

    def _reset_audio_buffer(self):
        self._audio_buffer = b""
        self._playback_started = False

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        self._reset_audio_buffer()
        if self._bot_speaking and self._client:
            logger.debug(
                f"{self}: interruption detected, sending input_text.done and waiting for speech.completed"
            )
            self._disconnecting = True
            self._speech_completed_event.clear()
            try:
                await self._client.send_json({"type": "input_text.done"})
                await asyncio.wait_for(self._speech_completed_event.wait(), timeout=5.0)
            except TimeoutError:
                logger.warning(f"{self}: timed out waiting for conversation.item.speech.completed")
        await super()._handle_interruption(frame, direction)

    async def _handle_audio_chunk(self, audio: bytes, context_id: str | None = None):
        """Buffer audio and emit frames using a jitter-buffer approach.

        Holds back audio until chunk_size bytes have been accumulated (to avoid
        glitches at the start of playback), then emits each subsequent chunk
        immediately as it arrives.
        """
        self._audio_buffer += audio

        if not self._playback_started:
            if len(self._audio_buffer) < self.chunk_size:
                return
            self._playback_started = True

        await self.push_frame(
            TTSAudioRawFrame(
                audio=self._audio_buffer,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )
        )
        self._audio_buffer = b""

    async def _receive_messages(self):
        """Receive NIM JSON events and push audio frames."""
        while self._client and self._client.is_active and not self._disconnecting:
            result = await self._client.receive_response()

            if self._disconnecting:
                self._speech_completed_event.set()

            if result is None:
                break

            if not (hasattr(result, "value") and hasattr(result.value, "bytes_")):  # type: ignore[union-attr]
                continue

            payload = result.value.bytes_  # type: ignore[union-attr]
            if not payload:
                continue

            context_id = self.get_active_audio_context_id()

            try:
                msg = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Unexpected binary frame — treat as raw PCM
                await self._handle_audio_chunk(payload, context_id)
                continue

            event_type = msg.get("type", "")

            if event_type != "conversation.item.speech.data":
                logger.debug(f"{self}: received event: {event_type}")

            if event_type == "conversation.item.speech.data":
                chunk_b64 = msg.get("audio", "")
                if chunk_b64:
                    await self.stop_ttfb_metrics()
                    await self._handle_audio_chunk(base64.b64decode(chunk_b64), context_id)
            elif event_type == "error":
                await self.push_error(error_msg=f"NIM error: {msg.get('message', msg)}")
                # In case of error we need to reconnect, otherwise we are not going to receive audio from the TTS service anymore
                break
            elif event_type == "conversation.item.speech.completed":
                # Need to reconnect to reset the synthesis state and be able to synthesize new text
                break

            # synthesize_session.updated, input_text.committed, etc. are ignored.

    async def _send_session_config(self):
        """Send synthesize_session.update to configure voice and audio params."""
        logger.debug(f"{self}: sending session config, sample_rate={self.sample_rate}")
        assert self._client is not None
        await self._client.send_json(
            {
                "type": "synthesize_session.update",
                "session": {
                    "input_text_synthesis": {
                        "voice_name": self._settings.voice,
                        "language_code": self._settings.language,
                    },
                    "output_audio_params": {
                        "sample_rate_hz": self.sample_rate,
                    },
                },
            }
        )

    # ── Synthesis ─────────────────────────────────────────────────────────────

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send text to NIM; audio arrives asynchronously via _receive_messages."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        text = text.strip()
        if not text or not any(c.isalnum() for c in text):
            return

        try:
            if not self._client or not self._client.is_active:
                await self._connect()

            assert self._client is not None
            await self._client.send_json({"type": "input_text.append", "text": text})
            await self._client.send_json({"type": "input_text.commit"})
            await self.start_tts_usage_metrics(text)
            yield None
        except Exception as e:
            logger.error(f"{self}: TTS error: {e}")
            yield ErrorFrame(error=f"NvidiaSageMakerTTSService error: {e}")
