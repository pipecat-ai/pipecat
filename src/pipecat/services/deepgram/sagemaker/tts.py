#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram text-to-speech service for AWS SageMaker.

This module provides a Pipecat TTS service that connects to Deepgram models
deployed on AWS SageMaker endpoints. Uses HTTP/2 bidirectional streaming for
low-latency real-time speech synthesis with support for interruptions and
streaming audio output.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.aws.sagemaker.bidi_client import SageMakerBidiClient
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class DeepgramSageMakerTTSSettings(TTSSettings):
    """Settings for Deepgram SageMaker TTS service.

    Parameters:
        encoding: Audio encoding format (e.g. "linear16").
    """

    encoding: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class DeepgramSageMakerTTSService(TTSService):
    """Deepgram text-to-speech service for AWS SageMaker.

    Provides real-time speech synthesis using Deepgram models deployed on
    AWS SageMaker endpoints. Uses HTTP/2 bidirectional streaming for low-latency
    audio generation with support for interruptions via the Clear message.

    Requirements:

    - AWS credentials configured (via environment variables, AWS CLI, or instance metadata)
    - A deployed SageMaker endpoint with Deepgram TTS model: https://developers.deepgram.com/docs/deploy-amazon-sagemaker
    - ``pipecat-ai[sagemaker]`` installed

    Example::

        tts = DeepgramSageMakerTTSService(
            endpoint_name="my-deepgram-tts-endpoint",
            region="us-east-2",
            voice="aura-2-helena-en",
        )
    """

    _settings: DeepgramSageMakerTTSSettings

    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str,
        voice: str = "aura-2-helena-en",
        sample_rate: Optional[int] = None,
        encoding: str = "linear16",
        **kwargs,
    ):
        """Initialize the Deepgram SageMaker TTS service.

        Args:
            endpoint_name: Name of the SageMaker endpoint with Deepgram TTS model
                deployed (e.g., "my-deepgram-tts-endpoint").
            region: AWS region where the endpoint is deployed (e.g., "us-east-2").
            voice: Voice model to use for synthesis. Defaults to "aura-2-helena-en".
            sample_rate: Audio sample rate in Hz. If None, uses the value from StartFrame.
            encoding: Audio encoding format. Defaults to "linear16".
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(
            sample_rate=sample_rate,
            push_stop_frames=True,
            pause_frame_processing=True,
            append_trailing_space=True,
            settings=DeepgramSageMakerTTSSettings(
                model=voice,
                voice=voice,
                language=None,
                encoding=encoding,
            ),
            **kwargs,
        )

        self._endpoint_name = endpoint_name
        self._region = region

        self._client: Optional[SageMakerBidiClient] = None
        self._response_task: Optional[asyncio.Task] = None
        self._context_id: Optional[str] = None
        self._ttfb_started: bool = False

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Deepgram SageMaker TTS service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Deepgram SageMaker TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Deepgram SageMaker TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Deepgram SageMaker TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with special handling for LLM response end.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._ttfb_started = False

    async def _connect(self):
        """Connect to the SageMaker endpoint and start the BiDi session.

        Builds the Deepgram TTS query string, creates the BiDi client,
        starts the streaming session, and launches a background task for processing
        responses.
        """
        logger.debug("Connecting to Deepgram TTS on SageMaker...")

        query_string = (
            f"model={self._settings.voice}&encoding={self._settings.encoding}"
            f"&sample_rate={self.sample_rate}"
        )

        self._client = SageMakerBidiClient(
            endpoint_name=self._endpoint_name,
            region=self._region,
            model_invocation_path="v1/speak",
            model_query_string=query_string,
        )

        try:
            await self._client.start_session()

            self._response_task = self.create_task(self._process_responses())

            logger.debug("Connected to Deepgram TTS on SageMaker")
            await self._call_event_handler("on_connected")

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect(self):
        """Disconnect from the SageMaker endpoint.

        Sends a Close message to Deepgram, cancels the response processing task,
        and closes the BiDi session. Safe to call multiple times.
        """
        if self._client and self._client.is_active:
            logger.debug("Disconnecting from Deepgram TTS on SageMaker...")

            try:
                await self._client.send_json({"type": "Close"})
            except Exception as e:
                logger.warning(f"Failed to send Close message: {e}")

            if self._response_task and not self._response_task.done():
                await self.cancel_task(self._response_task)

            await self._client.close_session()

            logger.debug("Disconnected from Deepgram TTS on SageMaker")
            await self._call_event_handler("on_disconnected")

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if necessary.

        Since all settings are part of the SageMaker session query string,
        any setting change requires reconnecting to apply the new values.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        # Deepgram uses voice as the model, so keep them in sync for metrics
        if "voice" in changed:
            self._settings.model = self._settings.voice
            self._sync_model_name_to_metrics()

        # TODO: someday we could reconnect here to apply updated settings.
        # Code might look something like the below:
        # await self._disconnect()
        # await self._connect()

        self._warn_unhandled_updated_settings(changed)

        return changed

    async def _process_responses(self):
        """Process streaming responses from Deepgram TTS on SageMaker.

        Continuously receives responses from the BiDi stream. Attempts to decode
        each payload as UTF-8 JSON for control messages (Flushed, Cleared, Metadata,
        Warning). If decoding fails, treats the payload as raw audio bytes and pushes
        a TTSAudioRawFrame downstream.
        """
        try:
            while self._client and self._client.is_active:
                result = await self._client.receive_response()

                if result is None:
                    break

                if hasattr(result, "value") and hasattr(result.value, "bytes_"):
                    if result.value.bytes_:
                        payload = result.value.bytes_

                        # Try to decode as JSON control message first
                        try:
                            response_data = payload.decode("utf-8")
                            parsed = json.loads(response_data)
                            msg_type = parsed.get("type")

                            if msg_type == "Metadata":
                                logger.trace(f"Received metadata: {parsed}")
                            elif msg_type == "Flushed":
                                logger.trace(f"Received Flushed: {parsed}")
                            elif msg_type == "Cleared":
                                logger.trace(f"Received Cleared: {parsed}")
                            elif msg_type == "Warning":
                                logger.warning(
                                    f"{self} warning: "
                                    f"{parsed.get('description', 'Unknown warning')}"
                                )
                            else:
                                logger.debug(f"Received unknown message type: {parsed}")

                        except (UnicodeDecodeError, json.JSONDecodeError):
                            # Not JSON — treat as raw audio bytes
                            await self.stop_ttfb_metrics()
                            frame = TTSAudioRawFrame(
                                payload,
                                self.sample_rate,
                                1,
                                context_id=self._context_id,
                            )
                            await self.push_frame(frame)

        except asyncio.CancelledError:
            logger.debug("TTS response processor cancelled")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            logger.debug("TTS response processor stopped")

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruption by sending Clear message to Deepgram.

        The Clear message will clear Deepgram's internal text buffer and stop
        sending audio, allowing for a new response to be generated.
        """
        await super()._handle_interruption(frame, direction)
        self._ttfb_started = False

        if self._client and self._client.is_active:
            try:
                await self._client.send_json({"type": "Clear"})
            except Exception as e:
                logger.error(f"{self} error sending Clear message: {e}")

    async def flush_audio(self):
        """Flush any pending audio synthesis by sending Flush command.

        This should be called when the LLM finishes a complete response to force
        generation of audio from Deepgram's internal text buffer.
        """
        if self._client and self._client.is_active:
            try:
                await self._client.send_json({"type": "Flush"})
            except Exception as e:
                logger.error(f"{self} error sending Flush message: {e}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Deepgram TTS on SageMaker.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: TTSStartedFrame, then None (audio comes asynchronously via
            the response processor).
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._ttfb_started:
                await self.start_ttfb_metrics()
                self._ttfb_started = True
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame(context_id=context_id)
            self._context_id = context_id

            await self._client.send_json({"type": "Speak", "text": text})

            yield None

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
