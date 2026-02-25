#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Riva text-to-speech service implementation.

This module provides integration with NVIDIA Riva's TTS services through
gRPC API for high-quality speech synthesis.
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Generator, Mapping, Optional

from pipecat.utils.tracing.service_decorators import traced_tts

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

try:
    import riva.client
    import riva.client.proto.riva_tts_pb2 as rtts
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use NVIDIA Riva TTS, you need to `pip install pipecat-ai[nvidia]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class NvidiaTTSSettings(TTSSettings):
    """Settings for NVIDIA Riva TTS service.

    Parameters:
        quality: Audio quality setting (0-100).
    """

    quality: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class NvidiaTTSService(TTSService):
    """NVIDIA Riva text-to-speech service.

    Provides high-quality text-to-speech synthesis using NVIDIA Riva's
    cloud-based TTS models. Supports multiple voices, languages, and
    configurable quality settings.
    """

    _settings: NvidiaTTSSettings

    class InputParams(BaseModel):
        """Input parameters for Riva TTS configuration.

        Parameters:
            language: Language code for synthesis. Defaults to US English.
            quality: Audio quality setting (0-100). Defaults to 20.
        """

        language: Optional[Language] = Language.EN_US
        quality: Optional[int] = 20

    def __init__(
        self,
        *,
        api_key: str,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice_id: str = "Magpie-Multilingual.EN-US.Aria",
        sample_rate: Optional[int] = None,
        model_function_map: Mapping[str, str] = {
            "function_id": "877104f7-e885-42b9-8de8-f6e4c6303969",
            "model_name": "magpie-tts-multilingual",
        },
        params: Optional[InputParams] = None,
        use_ssl: bool = True,
        **kwargs,
    ):
        """Initialize the NVIDIA Riva TTS service.

        Args:
            api_key: NVIDIA API key for authentication.
            server: gRPC server endpoint. Defaults to NVIDIA's cloud endpoint.
            voice_id: Voice model identifier. Defaults to multilingual Ray voice.
            sample_rate: Audio sample rate. If None, uses service default.
            model_function_map: Dictionary containing function_id and model_name for the TTS model.
            params: Additional configuration parameters for TTS synthesis.
            use_ssl: Whether to use SSL for the NVIDIA Riva server. Defaults to True.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        params = params or NvidiaTTSService.InputParams()

        super().__init__(
            sample_rate=sample_rate,
            settings=NvidiaTTSSettings(
                model=model_function_map.get("model_name"),
                voice=voice_id,
                language=params.language,
                quality=params.quality,
            ),
            **kwargs,
        )

        self._server = server
        self._api_key = api_key
        self._function_id = model_function_map.get("function_id")
        self._use_ssl = use_ssl

        self._service = None
        self._config = None

    async def set_model(self, model: str):
        """Set the TTS model.

        .. deprecated:: 0.0.104
            Model cannot be changed after initialization for NVIDIA Riva TTS.
            Set model and function id in the constructor instead, e.g.::

                NvidiaTTSService(
                    api_key=...,
                    model_function_map={"function_id": "<UUID>", "model_name": "<model_name>"},
                )

        Args:
            model: The model name to set.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "'set_model' is deprecated. Model cannot be changed after initialization"
                " for NVIDIA Riva TTS. Set model and function id in the constructor"
                " instead, e.g.: NvidiaTTSService(api_key=..., model_function_map="
                "{'function_id': '<UUID>', 'model_name': '<model_name>'})",
                DeprecationWarning,
                stacklevel=2,
            )

    async def _update_settings(self, delta: NvidiaTTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Settings are stored but not applied to the active connection.
        """
        changed = await super()._update_settings(delta)
        if not changed:
            return changed
        # TODO: reconnect gRPC client to apply changed settings.
        self._warn_unhandled_updated_settings(changed)
        return changed

    def _initialize_client(self):
        if self._service is not None:
            return

        metadata = [
            ["function-id", self._function_id],
            ["authorization", f"Bearer {self._api_key}"],
        ]
        auth = riva.client.Auth(None, self._use_ssl, self._server, metadata)

        self._service = riva.client.SpeechSynthesisService(auth)

    def _create_synthesis_config(self):
        if not self._service:
            return

        # warm up the service
        config = self._service.stub.GetRivaSynthesisConfig(
            riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest()
        )
        return config

    async def start(self, frame: StartFrame):
        """Start the Cartesia TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._initialize_client()
        self._config = self._create_synthesis_config()
        logger.debug(f"Initialized NvidiaTTSService with model: {self._settings.model}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using NVIDIA Riva TTS.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech data.
        """

        def read_audio_responses() -> Generator[rtts.SynthesizeSpeechResponse, None, None]:
            responses = self._service.synthesize_online(
                text,
                self._settings.voice,
                self._settings.language,
                sample_rate_hz=self.sample_rate,
                zero_shot_audio_prompt_file=None,
                zero_shot_quality=self._settings.quality,
                custom_dictionary={},
            )
            return responses

        def async_next(it):
            try:
                return next(it)
            except StopIteration:
                return None

        async def async_iterator(iterator) -> AsyncIterator[rtts.SynthesizeSpeechResponse]:
            while True:
                item = await asyncio.to_thread(async_next, iterator)
                if item is None:
                    return
                yield item

        try:
            assert self._service is not None, "TTS service not initialized"
            assert self._config is not None, "Synthesis configuration not created"

            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)

            logger.debug(f"{self}: Generating TTS [{text}]")

            responses = await asyncio.to_thread(read_audio_responses)

            async for resp in async_iterator(responses):
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=resp.audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
                yield frame

            await self.start_tts_usage_metrics(text)
            yield TTSStoppedFrame(context_id=context_id)
        except asyncio.TimeoutError as e:
            logger.error(f"{self} timeout waiting for audio response")
            yield ErrorFrame(error=f"{self} error: {e}")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")
