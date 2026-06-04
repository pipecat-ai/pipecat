#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Nemotron ASR STT service backed by an AWS SageMaker bidirectional-stream endpoint.

Uses SageMaker's HTTP/2 bidi-stream API to maintain a persistent connection to
the wrapper's /invocations-bidirectional-stream endpoint, which proxies to NIM's
realtime WebSocket.

Audio is streamed as base64-encoded PCM16 chunks via input_audio_buffer.append
events.  Transcription deltas arrive as InterimTranscriptionFrames and final
results as TranscriptionFrames.

When the VAD detects the user has stopped speaking, input_audio_buffer.commit
is sent to trigger NIM to finalise the current utterance.
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.aws.sagemaker.bidi_client import SageMakerBidiClient
from pipecat.services.settings import STTSettings, assert_given
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


@dataclass
class NvidiaSageMakerSTTSettings(STTSettings):
    """Settings for NvidiaSageMakerSTTService.

    Parameters:
        language: ISO-639-1 language code passed to NIM (e.g. ``en-US``).
    """


class NvidiaSageMakerSTTService(STTService):
    """NVIDIA Nemotron ASR STT service using SageMaker bidirectional streaming.

    Maintains a persistent HTTP/2 bidi-stream session to the SageMaker endpoint
    for the lifetime of the pipeline.  Audio chunks are forwarded as base64-encoded
    PCM16 via NIM realtime events; transcription results arrive asynchronously and
    are pushed as :class:`InterimTranscriptionFrame` and :class:`TranscriptionFrame`
    frames.

    Example::

        stt = NvidiaSageMakerSTTService(
            endpoint_name=os.getenv("SAGEMAKER_ASR_ENDPOINT_NAME"),
            region=os.getenv("AWS_REGION", "us-west-2"),
            settings=NvidiaSageMakerSTTService.Settings(
                language="en-US",
            ),
        )
    """

    Settings = NvidiaSageMakerSTTSettings

    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str = "us-west-2",
        sample_rate: int | None = None,
        settings: NvidiaSageMakerSTTSettings | None = None,
        ttfs_p99_latency: float | None = 1.5,
        **kwargs,
    ):
        """Initialize the SageMaker WebSocket STT service.

        Args:
            endpoint_name: Name of the deployed SageMaker endpoint.
            region: AWS region where the endpoint lives.
            sample_rate: Input sample rate in Hz. Defaults to pipeline rate.
            settings: Runtime-updatable settings (language, model).
            ttfs_p99_latency: Expected p99 time-to-first-segment latency in seconds.
            **kwargs: Forwarded to :class:`STTService`.
        """
        default_settings = self.Settings(
            model="cache-aware-parakeet-rnnt-en-US-asr-streaming-sortformer",
            language="en-US",
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        self._endpoint_name = endpoint_name
        self._region = region
        self._client: SageMakerBidiClient | None = None
        self._response_task: asyncio.Task | None = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics generation.
        """
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, frame: StartFrame):
        """Start the STT service and connect to the SageMaker endpoint.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and disconnect from the SageMaker endpoint.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and disconnect from the SageMaker endpoint.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    # ── Audio input ───────────────────────────────────────────────────────────

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send an audio chunk to NIM; transcription results arrive asynchronously.

        Each chunk is appended and immediately committed, matching the NVIDIA
        reference client pattern for continuous streaming transcription.
        """
        if self._client and self._client.is_active:
            try:
                await self._client.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio).decode(),
                    }
                )
                await self._client.send_json({"type": "input_audio_buffer.commit"})
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
        yield None

    # ── VAD integration ───────────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with VAD-specific handling for metrics lifecycle.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            logger.debug(f"{self}: VAD user started speaking")
            await self.start_processing_metrics()
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            logger.debug(f"{self}: VAD user stopped speaking")

    # ── Connection management ─────────────────────────────────────────────────

    async def _open_client_session(self):
        self._client = SageMakerBidiClient(
            endpoint_name=self._endpoint_name,
            region=self._region,
            model_query_string=None,
            model_invocation_path=None,
        )
        await self._client.start_session()
        await self._send_session_config()

    async def _close_client_session(self):
        if self._client and self._client.is_active:
            try:
                await self._client.send_json({"type": "session.end"})
            except Exception as e:
                logger.warning(f"{self}: error sending session.end: {e}")
            await self._client.close_session()
        self._client = None

    async def _connect(self):
        logger.debug(
            f"{self}: connecting to SageMaker bidi-stream endpoint '{self._endpoint_name}'"
        )
        try:
            await self._open_client_session()
            self._response_task = self.create_task(self._process_responses())
            logger.debug(f"{self}: connected")
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self}: connection error: {e}")
            self._client = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect(self):
        if self._response_task and not self._response_task.done():
            await self.cancel_task(self._response_task)
            self._response_task = None
        await self._close_client_session()
        await self._call_event_handler("on_disconnected")

    async def _do_reconnect(self):
        await self._close_client_session()
        await self._open_client_session()

    async def _send_session_config(self):
        """Send transcription_session.update to configure audio format and params.

        Specifies ``"model": "nemotron-asr-streaming"`` in ``input_audio_transcription`` so
        NIM selects the correct Nemotron ASR Streaming model.
        """
        logger.debug(
            f"{self}: sending session config,"
            f" sample_rate={self.sample_rate} language={self._settings.language}"
        )
        assert self._client is not None
        await self._client.send_json(
            {
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_params": {
                        "sample_rate_hz": self.sample_rate,
                        "num_channels": 1,
                    },
                    "input_audio_transcription": {
                        "language": self._settings.language,
                        "model": self._settings.model,
                    },
                    "recognition_config": {
                        "enable_automatic_punctuation": True,
                    },
                },
            }
        )

    # ── Response processing ───────────────────────────────────────────────────

    async def _process_responses(self):
        """Receive NIM JSON events and push transcription frames."""
        try:
            while self._client and self._client.is_active:
                result = await self._client.receive_response()

                if result is None or not (
                    hasattr(result, "value") and hasattr(result.value, "bytes_")  # type: ignore[union-attr]
                ):
                    continue

                payload = result.value.bytes_  # type: ignore[union-attr]
                if not payload:
                    continue

                try:
                    msg = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue

                event_type = msg.get("type", "")

                if event_type not in (
                    "conversation.item.input_audio_transcription.delta",
                    "input_audio_buffer.committed",
                ):
                    logger.debug(f"{self}: received event: {event_type}")

                _lang = assert_given(self._settings.language)
                language: Language | None = Language(_lang) if _lang is not None else None

                if event_type == "conversation.item.input_audio_transcription.delta":
                    delta = msg.get("delta", "")
                    if delta:
                        logger.debug(f"{self}: received transcription delta: {delta}")
                        await self.push_frame(
                            InterimTranscriptionFrame(
                                delta,
                                self._user_id,
                                time_now_iso8601(),
                                language=language,
                                result=msg,
                            )
                        )

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = msg.get("transcript", "")
                    if transcript.strip():
                        logger.debug(f"{self}: received final transcription: {transcript}")
                        await self.push_frame(
                            TranscriptionFrame(
                                transcript,
                                self._user_id,
                                time_now_iso8601(),
                                language=language,
                                result=msg,
                                finalized=True,
                            )
                        )
                        await self._handle_transcription(transcript, True)
                        await self.stop_processing_metrics()

                elif event_type in (
                    "conversation.item.input_audio_transcription.failed",
                    "error",
                ):
                    await self.push_error(error_msg=f"NIM ASR error: {msg}")
                    # In case of error we need to reconnect, otherwise we are not going to receive from the STT service anymore
                    await self._request_reconnect()

        except asyncio.CancelledError:
            logger.debug(f"{self}: response processor cancelled")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            logger.debug(f"{self}: response processor stopped")

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language=None):
        pass
