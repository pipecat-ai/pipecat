import base64
import json
import uuid
from typing import AsyncGenerator, List, Optional, Union

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import AudioContextWordTTSService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts


class WavesHttpTTSService(TTSService):
    class InputParams(BaseModel):
        language: str = "hi"
        speed: float = 1.2
        transliterate: bool = True
        remove_extra_silence: bool = True
        get_end_of_response_token: bool = True
        add_wav_header: bool = False
        consistency: Optional[float] = None
        similarity: Optional[float] = None
        enhancement: Optional[float] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "lightning",
        base_url: str = "https://waves-api.smallest.ai",
        aiohttp_sesssion: Optional[aiohttp.ClientSession] = None,
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._sample_rate = sample_rate
        self.set_voice(voice_id)
        self.set_model_name(model)
        self.base_url = base_url
        self.aiohttp_sesssion = aiohttp_sesssion or aiohttp.ClientSession()
        self._settings = {
            "language": params.language,
            "speed": params.speed,
            "transliterate": params.transliterate,
            "consistency": params.consistency,
            "similarity": params.similarity,
            "enhancement": params.enhancement,
            "get_end_of_response_token": params.get_end_of_response_token,
            "remove_extra_silence": params.remove_extra_silence,
            "add_wav_header": params.add_wav_header,
        }

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def _generate_audio(self, text: str) -> bytes:
        """Generate audio from text using the Waves API."""
        try:
            payload = {
                "voice_id": self._voice_id,
                "text": text,
                "sample_rate": self._sample_rate,
                **self._settings,
            }

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            url = f"{self.base_url}/api/v1/lightning/get_speech_long_text"
            async with self.aiohttp_sesssion.post(url, json=payload, headers=headers) as response:
                result = await response.read()

            return result
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            raise e

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_ttfb_metrics()

            audio = await self._generate_audio(text)

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            frame = TTSAudioRawFrame(audio=audio, sample_rate=self.sample_rate, num_channels=1)
            logger.debug(f"TTS audio frame received for text: {text}, chunk: {len(audio)}")

            yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()


class WavesSSETTSService(TTSService):
    class InputParams(BaseModel):
        language: str = "hi"
        speed: float = 1.2
        transliterate: bool = True
        remove_extra_silence: bool = True
        get_end_of_response_token: bool = True
        add_wav_header: bool = False
        consistency: Optional[float] = None
        similarity: Optional[float] = None
        enhancement: Optional[float] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "lightning-large",
        base_url: str = "https://waves-api.smallest.ai",
        aiohttp_sesssion: Optional[aiohttp.ClientSession] = None,
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._sample_rate = sample_rate
        self.set_voice(voice_id)
        self.set_model_name(model)
        self.base_url = base_url
        self.aiohttp_sesssion = aiohttp_sesssion or aiohttp.ClientSession()
        self._settings = {
            "language": params.language,
            "speed": params.speed,
            "transliterate": params.transliterate,
            "get_end_of_response_token": params.get_end_of_response_token,
            "remove_extra_silence": params.remove_extra_silence,
            "add_wav_header": params.add_wav_header,
        }

        if params.consistency is not None:
            self._settings["transliterate"] = params.transliterate

        if params.similarity is not None:
            self._settings["similarity"] = params.similarity

        if params.enhancement is not None:
            self._settings["enhancement"] = params.enhancement

        self._is_first_chunk = True

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        try:
            payload = {
                "voice_id": self._voice_id,
                "text": text,
                "sample_rate": self._sample_rate,
                **self._settings,
            }

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }

            await self.start_ttfb_metrics()
            url = f"{self.base_url}/api/v1/lightning-large/stream"
            async with self.aiohttp_sesssion.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.content:
                    line = line.strip()
                    if not line or line == b":" or line.startswith(b"event:"):
                        continue

                    if line.startswith(b"data:"):
                        data = line[5:].strip()  # Remove 'data:' prefix
                        if not data or data == b"[DONE]":
                            continue

                        try:
                            json_data = json.loads(data)
                            if "audio" in json_data:
                                audio_data = base64.b64decode(json_data["audio"])
                                if self._is_first_chunk:
                                    await self.stop_ttfb_metrics()
                                    yield TTSStartedFrame()
                                    self._is_first_chunk = False
                                frame = TTSAudioRawFrame(
                                    audio=audio_data, sample_rate=self.sample_rate, num_channels=1
                                )
                                yield frame
                        except Exception as e:
                            logger.error(f"Error processing audio chunk: {e}")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
        finally:
            yield TTSStoppedFrame()
