#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import struct
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic.main import BaseModel

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService
from pipecat.transcriptions.language import Language

try:
    from pyht.async_client import AsyncClient
    from pyht.client import TTSOptions
    from pyht.protos.api_pb2 import Format
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use PlayHT, you need to `pip install pipecat-ai[playht]`. Also, set `PLAY_HT_USER_ID` and `PLAY_HT_API_KEY` environment variables."
    )
    raise Exception(f"Missing module: {e}")


class PlayHTTTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0
        seed: Optional[int] = None

    def __init__(
        self,
        *,
        api_key: str,
        user_id: str,
        voice_url: str,
        voice_engine: str = "PlayHT3.0-mini",
        sample_rate: int = 16000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._user_id = user_id
        self._speech_key = api_key

        self._client = AsyncClient(
            user_id=self._user_id,
            api_key=self._speech_key,
        )
        self._settings = {
            "sample_rate": sample_rate,
            "language": self.language_to_service_language(params.language)
            if params.language
            else Language.EN,
            "format": Format.FORMAT_WAV,
            "voice_engine": voice_engine,
            "speed": params.speed,
            "seed": params.seed,
        }
        self.set_model_name(voice_engine)
        self.set_voice(voice_url)
        self._options = TTSOptions(
            voice=self._voice_id,
            language=self._settings["language"],
            sample_rate=self._settings["sample_rate"],
            format=self._settings["format"],
            speed=self._settings["speed"],
            seed=self._settings["seed"],
        )

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        match language:
            case Language.BG:
                return "BULGARIAN"
            case Language.CA:
                return "CATALAN"
            case Language.CS:
                return "CZECH"
            case Language.DA:
                return "DANISH"
            case Language.DE:
                return "GERMAN"
            case (
                Language.EN
                | Language.EN_US
                | Language.EN_GB
                | Language.EN_AU
                | Language.EN_NZ
                | Language.EN_IN
            ):
                return "ENGLISH"
            case Language.ES:
                return "SPANISH"
            case Language.FR | Language.FR_CA:
                return "FRENCH"
            case Language.EL:
                return "GREEK"
            case Language.HI:
                return "HINDI"
            case Language.HU:
                return "HUNGARIAN"
            case Language.ID:
                return "INDONESIAN"
            case Language.IT:
                return "ITALIAN"
            case Language.JA:
                return "JAPANESE"
            case Language.KO:
                return "KOREAN"
            case Language.MS:
                return "MALAY"
            case Language.NL:
                return "DUTCH"
            case Language.PL:
                return "POLISH"
            case Language.PT | Language.PT_BR:
                return "PORTUGUESE"
            case Language.RU:
                return "RUSSIAN"
            case Language.SV:
                return "SWEDISH"
            case Language.TH:
                return "THAI"
            case Language.TR:
                return "TURKISH"
            case Language.UK:
                return "UKRAINIAN"
        return None

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            b = bytearray()
            in_header = True

            await self.start_ttfb_metrics()

            playht_gen = self._client.tts(
                text, voice_engine=self._settings["voice_engine"], options=self._options
            )

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()
            async for chunk in playht_gen:
                # skip the RIFF header.
                if in_header:
                    b.extend(chunk)
                    if len(b) <= 36:
                        continue
                    else:
                        fh = io.BytesIO(b)
                        fh.seek(36)
                        (data, size) = struct.unpack("<4sI", fh.read(8))
                        while data != b"data":
                            fh.read(size)
                            (data, size) = struct.unpack("<4sI", fh.read(8))
                        in_header = False
                else:
                    if len(chunk):
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, self._settings["sample_rate"], 1)
                        yield frame
            yield TTSStoppedFrame()
        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
