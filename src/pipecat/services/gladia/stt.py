#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import warnings
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.ai_services import STTService
from pipecat.services.gladia.config import GladiaInputParams
from pipecat.services.gladia.language_mapping import language_to_gladia_language
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Gladia, you need to `pip install pipecat-ai[gladia]`.")
    raise Exception(f"Missing module: {e}")


# Deprecation warning for nested InputParams
class _InputParamsDescriptor:
    """Descriptor for backward compatibility with deprecation warning."""

    def __get__(self, obj, objtype=None):
        warnings.warn(
            "GladiaSTTService.InputParams is deprecated and will be removed in a future version. "
            "Import and use GladiaInputParams directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return GladiaInputParams


class GladiaSTTService(STTService):
    """Speech-to-Text service using Gladia's API.

    This service connects to Gladia's WebSocket API for real-time transcription
    with support for multiple languages, custom vocabulary, and various processing options.

    For complete API documentation, see: https://docs.gladia.io/api-reference/v2/live/init
    """

    # Maintain backward compatibility
    InputParams = _InputParamsDescriptor()

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "https://api.gladia.io/v2/live",
        confidence: float = 0.5,
        sample_rate: Optional[int] = None,
        model: str = "fast",
        params: GladiaInputParams = GladiaInputParams(),
        **kwargs,
    ):
        """Initialize the Gladia STT service.

        Args:
            api_key: Gladia API key
            url: Gladia API URL
            confidence: Minimum confidence threshold for transcriptions
            sample_rate: Audio sample rate in Hz
            model: Model to use ("fast" or "accurate")
            params: Additional configuration parameters
            **kwargs: Additional arguments passed to the STTService
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Warn about deprecated language parameter if it's used
        if params.language is not None:
            warnings.warn(
                "The 'language' parameter is deprecated and will be removed in a future version. "
                "Use 'language_config' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._api_key = api_key
        self._url = url
        self.set_model_name(model)
        self._confidence = confidence
        self._params = params
        self._websocket = None
        self._receive_task = None

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert pipecat Language enum to Gladia's language code."""
        return language_to_gladia_language(language)

    def _prepare_settings(self) -> Dict[str, Any]:
        settings = {
            "encoding": self._params.encoding or "wav/pcm",
            "bit_depth": self._params.bit_depth or 16,
            "sample_rate": self.sample_rate,
            "channels": self._params.channels or 1,
            "model": self._model_name,
        }

        # Add custom_metadata if provided
        if self._params.custom_metadata:
            settings["custom_metadata"] = self._params.custom_metadata

        # Add endpointing parameters if provided
        if self._params.endpointing is not None:
            settings["endpointing"] = self._params.endpointing
        if self._params.maximum_duration_without_endpointing is not None:
            settings["maximum_duration_without_endpointing"] = (
                self._params.maximum_duration_without_endpointing
            )

        # Add language configuration (prioritize language_config over deprecated language)
        if self._params.language_config:
            settings["language_config"] = self._params.language_config.model_dump(exclude_none=True)
        elif self._params.language:  # Backward compatibility for deprecated parameter
            language_code = self.language_to_service_language(self._params.language)
            if language_code:
                settings["language_config"] = {
                    "languages": [language_code],
                    "code_switching": False,
                }

        # Add pre_processing configuration if provided
        if self._params.pre_processing:
            settings["pre_processing"] = self._params.pre_processing.model_dump(exclude_none=True)

        # Add realtime_processing configuration if provided
        if self._params.realtime_processing:
            settings["realtime_processing"] = self._params.realtime_processing.model_dump(
                exclude_none=True
            )

        # Add messages_config if provided
        if self._params.messages_config:
            settings["messages_config"] = self._params.messages_config.model_dump(exclude_none=True)

        return settings

    async def start(self, frame: StartFrame):
        """Start the Gladia STT websocket connection."""
        await super().start(frame)
        if self._websocket:
            return
        settings = self._prepare_settings()
        response = await self._setup_gladia(settings)
        self._websocket = await websockets.connect(response["url"])
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler())

    async def stop(self, frame: EndFrame):
        """Stop the Gladia STT websocket connection."""
        await super().stop(frame)
        await self._send_stop_recording()
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        if self._receive_task:
            await self.wait_for_task(self._receive_task)
            self._receive_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the Gladia STT websocket connection."""
        await super().cancel(frame)
        await self._websocket.close()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run speech-to-text on audio data."""
        await self.start_processing_metrics()
        await self._send_audio(audio)
        await self.stop_processing_metrics()
        yield None

    async def _setup_gladia(self, settings: Dict[str, Any]):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url,
                headers={"X-Gladia-Key": self._api_key, "Content-Type": "application/json"},
                json=settings,
            ) as response:
                if response.ok:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Gladia error: {response.status}: {error_text or response.reason}"
                    )
                    raise Exception(
                        f"Failed to initialize Gladia session: {response.status} - {error_text}"
                    )

    async def _send_audio(self, audio: bytes):
        data = base64.b64encode(audio).decode("utf-8")
        message = {"type": "audio_chunk", "data": {"chunk": data}}
        await self._websocket.send(json.dumps(message))

    async def _send_stop_recording(self):
        if self._websocket and not self._websocket.closed:
            await self._websocket.send(json.dumps({"type": "stop_recording"}))

    async def _receive_task_handler(self):
        try:
            async for message in self._websocket:
                content = json.loads(message)
                if content["type"] == "transcript":
                    utterance = content["data"]["utterance"]
                    confidence = utterance.get("confidence", 0)
                    transcript = utterance["text"]
                    if confidence >= self._confidence:
                        if content["data"]["is_final"]:
                            await self.push_frame(
                                TranscriptionFrame(transcript, "", time_now_iso8601())
                            )
                        else:
                            await self.push_frame(
                                InterimTranscriptionFrame(transcript, "", time_now_iso8601())
                            )
        except websockets.exceptions.ConnectionClosed:
            # Expected when closing the connection
            pass
        except Exception as e:
            logger.error(f"Error in Gladia WebSocket handler: {e}")
