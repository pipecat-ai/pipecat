import base64
import json
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService
from pipecat.transcriptions.language import Language


def language_to_ola_krutrim_language(language: Language) -> Optional[str]:
    return None


class OlaKrutrimTTSService(TTSService):
    class InputParams(BaseModel):
        model_name: str = "fastpitch_tts"
        base_url: str = "https://cloud.olakrutrim.com/v1/audio/generations/tts"

    def __init__(self, *, api_key: str, params: InputParams = InputParams(), **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._model_name = params.model_name
        self._base_url = params.base_url
        self.set_model_name(params.model_name)

    async def set_model(self, model: str):
        logger.debug(f"Switching TTS model to: [{model}]")
        self._model_name = model
        await super().set_model(model)

    async def set_language(self, language: Language):
        pass

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")
        await self.push_frame(TTSStartedFrame())
        await self.start_ttfb_metrics()

        try:
            payload = {
                "modelName": self._model_name,
                "textToSpeak": text,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }
            url = f"{self._base_url}/{self._model_name}"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        await self.stop_ttfb_metrics()
                        resp_json = await resp.json()
                        output = resp_json["output"]
                        sr = resp_json["sampling_rate"]

                        op_audio = base64.b64decode(output)
                        op_audio = json.loads(op_audio)

                        frame = TTSAudioRawFrame(
                            audio=op_audio,
                            sample_rate=sr,
                            num_channels=1,
                        )
                        yield frame
                    else:
                        logger.error(f"Failed to get a response from the server. Status code: {resp.status}")
                        logger.error(f"Response content: {await resp.text()}")
                        raise RuntimeError("Failed to generate TTS")

        except aiohttp.ClientError as e:
            logger.error(f"HTTP Client Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected exception: {e}")

        await self.start_tts_usage_metrics(text)
        await self.push_frame(TTSStoppedFrame())

    async def stop(self, frame: EndFrame):
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
