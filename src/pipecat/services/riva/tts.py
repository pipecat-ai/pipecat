#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import AsyncGenerator, Optional

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

FASTPITCH_TIMEOUT_SECS = 5


class FastPitchTTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN_US
        quality: Optional[int] = 20

    def __init__(
        self,
        *,
        api_key: str,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice_id: str = "English-US.Female-1",
        sample_rate: Optional[int] = None,
        function_id: str = "0149dedb-2be8-4195-b9a0-e57e0e14f972",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._api_key = api_key
        self._voice_id = voice_id
        self._language_code = params.language
        self._quality = params.quality

        self.set_model_name("fastpitch-hifigan-tts")
        self.set_voice(voice_id)

        metadata = [
            ["function-id", function_id],
            ["authorization", f"Bearer {api_key}"],
        ]
        auth = riva.client.Auth(None, True, server, metadata)

        self._service = riva.client.SpeechSynthesisService(auth)

        # warm up the service
        config_response = self._service.stub.GetRivaSynthesisConfig(
            riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest()
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
                    audio_prompt_file=None,
                    quality=self._quality,
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
            resp = await asyncio.wait_for(queue.get(), FASTPITCH_TIMEOUT_SECS)
            while resp:
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=resp.audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                yield frame
                resp = await asyncio.wait_for(queue.get(), FASTPITCH_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            logger.error(f"{self} timeout waiting for audio response")

        await self.start_tts_usage_metrics(text)
        yield TTSStoppedFrame()
