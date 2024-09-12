#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import AsyncGenerator, Literal
from pydantic import BaseModel

from pipecat.frames.frames import AudioRawFrame, Frame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.ai_services import TTSService

from loguru import logger

# See .env.example for ElevenLabs configuration needed
try:
    from elevenlabs.client import AsyncElevenLabs
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use ElevenLabs, you need to `pip install pipecat-ai[elevenlabs]`. Also, set `ELEVENLABS_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


def sample_rate_from_output_format(output_format: str) -> int:
    match output_format:
        case "pcm_16000":
            return 16000
        case "pcm_22050":
            return 22050
        case "pcm_24000":
            return 24000
        case "pcm_44100":
            return 44100
    return 16000


class ElevenLabsTTSService(TTSService):
    class InputParams(BaseModel):
        output_format: Literal["pcm_16000", "pcm_22050", "pcm_24000", "pcm_44100"] = "pcm_16000"

    def __init__(
            self,
            *,
            api_key: str,
            voice_id: str,
            model: str = "eleven_turbo_v2_5",
            params: InputParams = InputParams(),
            **kwargs):
        super().__init__(**kwargs)

        self._voice_id = voice_id
        self._model = model
        self._params = params
        self._client = AsyncElevenLabs(api_key=api_key)
        self._sample_rate = sample_rate_from_output_format(params.output_format)

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        logger.debug(f"Switching TTS model to: [{model}]")
        self._model = model

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice_id = voice

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        await self.start_tts_usage_metrics(text)
        await self.start_ttfb_metrics()

        results = await self._client.generate(
            text=text,
            voice=self._voice_id,
            model=self._model,
            output_format=self._params.output_format
        )

        tts_started = False
        async for audio in results:
            # This is so we send TTSStartedFrame when we have the first audio
            # bytes.
            if not tts_started:
                await self.push_frame(TTSStartedFrame())
                tts_started = True
            await self.stop_ttfb_metrics()
            frame = AudioRawFrame(audio, self._sample_rate, 1)
            yield frame

        await self.push_frame(TTSStoppedFrame())
