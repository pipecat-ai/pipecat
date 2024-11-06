#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic.main import BaseModel

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
        language_map = {
            Language.BG: "bg",
            Language.CA: "ca",
            Language.ZH: "zh",
            Language.CS: "cs",
            Language.DA: "da",
            Language.NL: "nl",
            Language.EN: "en",
            Language.EN_US: "en",
            Language.EN_AU: "en",
            Language.EN_GB: "en",
            Language.EN_NZ: "en",
            Language.EN_IN: "en",
            Language.ET: "et",
            Language.FI: "fi",
            Language.FR: "fr",
            Language.FR_CA: "fr",
            Language.DE: "de",
            Language.DE_CH: "de",
            Language.EL: "el",
            Language.HI: "hi",
            Language.HU: "hu",
            Language.ID: "id",
            Language.IT: "it",
            Language.JA: "ja",
            Language.KO: "ko",
            Language.LV: "lv",
            Language.LT: "lt",
            Language.MS: "ms",
            Language.NO: "no",
            Language.PL: "pl",
            Language.PT: "pt",
            Language.PT_BR: "pt",
            Language.RO: "ro",
            Language.RU: "ru",
            Language.SK: "sk",
            Language.ES: "es",
            Language.SV: "sv",
            Language.TH: "th",
            Language.TR: "tr",
            Language.UK: "uk",
            Language.VI: "vi",
        }
        return language_map.get(language)

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
