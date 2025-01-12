#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.ai_services import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Gladia, you need to `pip install pipecat-ai[gladia]`. Also, set `GLADIA_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


def language_to_gladia_language(language: Language) -> str | None:
    BASE_LANGUAGES = {
        Language.AF: "af",
        Language.AM: "am",
        Language.AR: "ar",
        Language.AS: "as",
        Language.AZ: "az",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.BS: "bs",
        Language.CA: "ca",
        Language.CS: "cs",
        Language.CY: "cy",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.ET: "et",
        Language.EU: "eu",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GA: "ga",
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.HY: "hy",
        Language.ID: "id",
        Language.IS: "is",
        Language.IT: "it",
        Language.JA: "ja",
        Language.JV: "jv",
        Language.KA: "ka",
        Language.KK: "kk",
        Language.KM: "km",
        Language.KN: "kn",
        Language.KO: "ko",
        Language.LO: "lo",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MN: "mn",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.MT: "mt",
        Language.MY: "my",
        Language.NE: "ne",
        Language.NL: "nl",
        Language.NO: "no",
        Language.OR: "or",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PS: "ps",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SI: "si",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SO: "so",
        Language.SQ: "sq",
        Language.SR: "sr",
        Language.SU: "su",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TH: "th",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.UZ: "uz",
        Language.VI: "vi",
        Language.ZH: "zh",
        Language.ZU: "zu",
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        # Convert enum value to string and get the base language part (e.g. es-ES -> es)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Look up the base code in our supported languages
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


class GladiaSTTService(STTService):
    class InputParams(BaseModel):
        sample_rate: Optional[int] = 16000
        language: Optional[Language] = Language.EN
        endpointing: Optional[float] = 0.2
        maximum_duration_without_endpointing: Optional[int] = 10
        audio_enhancer: Optional[bool] = None
        words_accurate_timestamps: Optional[bool] = None

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "https://api.gladia.io/v2/live",
        confidence: float = 0.5,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._api_key = api_key
        self._url = url
        self._settings = {
            "encoding": "wav/pcm",
            "bit_depth": 16,
            "sample_rate": params.sample_rate,
            "channels": 1,
            "language_config": {
                "languages": [self.language_to_service_language(params.language)]
                if params.language
                else [],
                "code_switching": False,
            },
            "endpointing": params.endpointing,
            "maximum_duration_without_endpointing": params.maximum_duration_without_endpointing,
            "pre_processing": {
                "audio_enhancer": params.audio_enhancer,
            },
            "realtime_processing": {
                "words_accurate_timestamps": params.words_accurate_timestamps,
            },
        }
        self._confidence = confidence

    def language_to_service_language(self, language: Language) -> str | None:
        return language_to_gladia_language(language)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        response = await self._setup_gladia()
        self._websocket = await websockets.connect(response["url"])
        self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._send_stop_recording()
        await self._websocket.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._websocket.close()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self.start_processing_metrics()
        await self._send_audio(audio)
        await self.stop_processing_metrics()
        yield None

    async def _setup_gladia(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url,
                headers={"X-Gladia-Key": self._api_key, "Content-Type": "application/json"},
                json=self._settings,
            ) as response:
                if response.ok:
                    return await response.json()
                else:
                    logger.error(
                        f"Gladia error: {response.status}: {response.text or response.reason}"
                    )
                    raise Exception(f"Failed to initialize Gladia session: {response.status}")

    async def _send_audio(self, audio: bytes):
        data = base64.b64encode(audio).decode("utf-8")
        message = {"type": "audio_chunk", "data": {"chunk": data}}
        await self._websocket.send(json.dumps(message))

    async def _send_stop_recording(self):
        await self._websocket.send(json.dumps({"type": "stop_recording"}))

    async def _receive_task_handler(self):
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
