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
from typing import AsyncGenerator, AsyncIterator, Generator, Mapping, Optional

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
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

try:
    import riva.client
    import riva.client.proto.riva_tts_pb2 as rtts
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use NVIDIA Riva TTS, you need to `pip install pipecat-ai[nvidia]`.")
    raise Exception(f"Missing module: {e}")


class NvidiaTTSService(TTSService):
    """NVIDIA Riva text-to-speech service.

    Provides high-quality text-to-speech synthesis using NVIDIA Riva's
    cloud-based TTS models. Supports multiple voices, languages, and
    configurable quality settings.
    """

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
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or NvidiaTTSService.InputParams()

        self._server = server
        self._api_key = api_key
        self._voice_id = voice_id
        self._language_code = params.language
        self._quality = params.quality
        self._function_id = model_function_map.get("function_id")
        self._use_ssl = use_ssl
        self.set_model_name(model_function_map.get("model_name"))
        self.set_voice(voice_id)

        self._service = None
        self._config = None

    async def set_model(self, model: str):
        """Attempt to set the TTS model.

        Note: Model cannot be changed after initialization for Riva service.

        Args:
            model: The model name to set (operation not supported).
        """
        logger.warning(f"Cannot set model after initialization. Set model and function id like so:")
        example = {"function_id": "<UUID>", "model_name": "<model_name>"}
        logger.warning(
            f"{self.__class__.__name__}(api_key=<api_key>, model_function_map={example})"
        )

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
        logger.debug(f"Initialized NvidiaTTSService with model: {self.model_name}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using NVIDIA Riva TTS.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech data.
        """

        def read_audio_responses() -> Generator[rtts.SynthesizeSpeechResponse, None, None]:
            responses = self._service.synthesize_online(
                text,
                self._voice_id,
                self._language_code,
                sample_rate_hz=self.sample_rate,
                zero_shot_audio_prompt_file=None,
                zero_shot_quality=self._quality,
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
            yield TTSStartedFrame()

            logger.debug(f"{self}: Generating TTS [{text}]")

            responses = await asyncio.to_thread(read_audio_responses)

            async for resp in async_iterator(responses):
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=resp.audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                yield frame

            await self.start_tts_usage_metrics(text)
            yield TTSStoppedFrame()
        except asyncio.TimeoutError:
            logger.error(f"{self} timeout waiting for audio response")
            yield ErrorFrame(error=f"{self} error: {e}")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")
