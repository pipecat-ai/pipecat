#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
from typing import AsyncGenerator, Mapping, Optional

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
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN_US
        quality: Optional[int] = 20

    def __init__(
        self,
        *,
        api_key: str = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice_id: str = "Magpie-Multilingual.EN-US.Ray",
        sample_rate: Optional[int] = None,
        model_function_map: Mapping[str, str] = {
            "function_id": "877104f7-e885-42b9-8de8-f6e4c6303969",
            "model_name": "magpie-tts-multilingual",
        },
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
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
        logger.warning(f"Cannot set model after initialization. Set model and function id like so:")
        example = {"function_id": "<UUID>", "model_name": "<model_name>"}
        logger.warning(
            f"{self.__class__.__name__}(api_key=<api_key>, model_function_map={example})"
        )

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
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
            resp = await asyncio.wait_for(queue.get(), RIVA_TTS_TIMEOUT_SECS)
            while resp:
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=resp.audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                yield frame
                resp = await asyncio.wait_for(queue.get(), RIVA_TTS_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            logger.error(f"{self} timeout waiting for audio response")

        await self.start_tts_usage_metrics(text)
        yield TTSStoppedFrame()


class FastPitchTTSService(RivaTTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN_US
        quality: Optional[int] = 20

    def __init__(
        self,
        *,
        api_key: str = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice_id: str = "English-US.Female-1",
        sample_rate: Optional[int] = None,
        model_function_map: Mapping[str, str] = {
            "function_id": "0149dedb-2be8-4195-b9a0-e57e0e14f972",
            "model_name": "fastpitch-hifigan-tts",
        },
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
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
