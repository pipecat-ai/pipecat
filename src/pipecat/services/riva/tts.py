#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Riva text-to-speech service implementation.

This module provides integration with NVIDIA Riva's TTS services through
gRPC API for high-quality speech synthesis.
"""

import asyncio
import os
from typing import AsyncGenerator, Mapping, Optional

from pipecat.utils.tracing.service_decorators import traced_tts

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

try:
    import riva.client

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use NVIDIA Riva TTS, you need to `pip install pipecat-ai[riva]`.")
    raise Exception(f"Missing module: {e}")

RIVA_TTS_TIMEOUT_SECS = 5


class RivaTTSService(TTSService):
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
        voice_id: str = "Magpie-Multilingual.EN-US.Ray",
        sample_rate: Optional[int] = None,
        model_function_map: Mapping[str, str] = {
            "function_id": "877104f7-e885-42b9-8de8-f6e4c6303969",
            "model_name": "magpie-tts-multilingual",
        },
        params: Optional[InputParams] = None,
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
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or RivaTTSService.InputParams()

        self._api_key = api_key
        self._voice_id = voice_id
        self._language_code = params.language
        self._quality = params.quality
        self._function_id = model_function_map.get("function_id")

        self.set_model_name(model_function_map.get("model_name"))
        self.set_voice(voice_id)

        metadata = [
            ["function-id", self._function_id],
            ["authorization", f"Bearer {api_key}"],
        ]
        auth = riva.client.Auth(None, True, server, metadata)

        self._service = riva.client.SpeechSynthesisService(auth)

        # warm up the service
        config_response = self._service.stub.GetRivaSynthesisConfig(
            riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest()
        )

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

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using NVIDIA Riva TTS.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech data.
        """

        def read_audio_responses(queue: asyncio.Queue):
            def add_response(r):
                asyncio.run_coroutine_threadsafe(queue.put(r), self.get_event_loop())

            try:
                responses = self._service.synthesize_online(
                    text,
                    self._voice_id,
                    self._language_code,
                    sample_rate_hz=self.sample_rate,
                    zero_shot_audio_prompt_file=None,
                    zero_shot_quality=self._quality,
                    custom_dictionary={},
                )
                for r in responses:
                    add_response(r)
                add_response(None)
            except Exception as e:
                logger.error(f"{self} exception: {e}")
                add_response(None)

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            queue = asyncio.Queue()
            await asyncio.to_thread(read_audio_responses, queue)

            # Wait for the thread to start.
            resp = await asyncio.wait_for(queue.get(), timeout=RIVA_TTS_TIMEOUT_SECS)
            while resp:
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=resp.audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                yield frame
                resp = await asyncio.wait_for(queue.get(), timeout=RIVA_TTS_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            logger.error(f"{self} timeout waiting for audio response")

        await self.start_tts_usage_metrics(text)
        yield TTSStoppedFrame()


class FastPitchTTSService(RivaTTSService):
    """Deprecated FastPitch TTS service.

    .. deprecated:: 0.0.66
        This class is deprecated. Use RivaTTSService instead for new implementations.
        Provides backward compatibility for existing FastPitch TTS integrations.
    """

    def __init__(
        self,
        *,
        api_key: str,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice_id: str = "English-US.Female-1",
        sample_rate: Optional[int] = None,
        model_function_map: Mapping[str, str] = {
            "function_id": "0149dedb-2be8-4195-b9a0-e57e0e14f972",
            "model_name": "fastpitch-hifigan-tts",
        },
        params: Optional[RivaTTSService.InputParams] = None,
        **kwargs,
    ):
        """Initialize the deprecated FastPitch TTS service.

        Args:
            api_key: NVIDIA API key for authentication.
            server: gRPC server endpoint. Defaults to NVIDIA's cloud endpoint.
            voice_id: Voice model identifier. Defaults to Female-1 voice.
            sample_rate: Audio sample rate. If None, uses service default.
            model_function_map: Dictionary containing function_id and model_name for FastPitch model.
            params: Additional configuration parameters for TTS synthesis.
            **kwargs: Additional arguments passed to parent RivaTTSService.
        """
        super().__init__(
            api_key=api_key,
            server=server,
            voice_id=voice_id,
            sample_rate=sample_rate,
            model_function_map=model_function_map,
            params=params,
            **kwargs,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "`FastPitchTTSService` is deprecated, use `RivaTTSService` instead.",
                DeprecationWarning,
            )
