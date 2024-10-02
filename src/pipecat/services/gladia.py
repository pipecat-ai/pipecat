#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from typing import AsyncGenerator, Optional

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

# See .env.example for Gladia configuration needed
try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Gladia, you need to `pip install pipecat-ai[gladia]`. Also, set `GLADIA_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


def language_to_gladia_language(language: Language) -> str | None:
    match language:
        case Language.BG:
            return "bg"
        case Language.CA:
            return "ca"
        case Language.ZH:
            return "zh"
        case Language.CS:
            return "cs"
        case Language.DA:
            return "da"
        case Language.NL:
            return "nl"
        case (
            Language.EN
            | Language.EN_US
            | Language.EN_AU
            | Language.EN_GB
            | Language.EN_NZ
            | Language.EN_IN
        ):
            return "en"
        case Language.ET:
            return "et"
        case Language.FI:
            return "fi"
        case Language.FR | Language.FR_CA:
            return "fr"
        case Language.DE | Language.DE_CH:
            return "de"
        case Language.EL:
            return "el"
        case Language.HI:
            return "hi"
        case Language.HU:
            return "hu"
        case Language.ID:
            return "id"
        case Language.IT:
            return "it"
        case Language.JA:
            return "ja"
        case Language.KO:
            return "ko"
        case Language.LV:
            return "lv"
        case Language.LT:
            return "lt"
        case Language.MS:
            return "ms"
        case Language.NO:
            return "no"
        case Language.PL:
            return "pl"
        case Language.PT | Language.PT_BR:
            return "pt"
        case Language.RO:
            return "ro"
        case Language.RU:
            return "ru"
        case Language.SK:
            return "sk"
        case Language.ES:
            return "es"
        case Language.SV:
            return "sv"
        case Language.TH:
            return "th"
        case Language.TR:
            return "tr"
        case Language.UK:
            return "uk"
        case Language.VI:
            return "vi"
    return None


class GladiaSTTService(STTService):
    class InputParams(BaseModel):
        sample_rate: Optional[int] = 16000
        language: Optional[Language] = Language.EN
        transcription_hint: Optional[str] = None
        endpointing: Optional[int] = 200
        prosody: Optional[bool] = None

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.gladia.io/audio/text/audio-transcription",
        confidence: float = 0.5,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._api_key = api_key
        self._url = url
        self._settings = {
            "sample_rate": params.sample_rate,
            "language": language_to_gladia_language(params.language) if params.language else "en",
            "transcription_hint": params.transcription_hint,
            "endpointing": params.endpointing,
            "prosody": params.prosody,
        }
        self._confidence = confidence

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._websocket = await websockets.connect(self._url)
        self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
        await self._setup_gladia()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
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
        configuration = {
            "x_gladia_key": self._api_key,
            "encoding": "WAV/PCM",
            "model_type": "fast",
            "language_behaviour": "manual",
            "sample_rate": self._settings["sample_rate"],
            "language": self._settings["language"],
            "transcription_hint": self._settings["transcription_hint"],
            "endpointing": self._settings["endpointing"],
            "prosody": self._settings["prosody"],
        }

        await self._websocket.send(json.dumps(configuration))

    async def _send_audio(self, audio: bytes):
        message = {"frames": base64.b64encode(audio).decode("utf-8")}
        await self._websocket.send(json.dumps(message))

    async def _receive_task_handler(self):
        async for message in self._websocket:
            utterance = json.loads(message)
            if not utterance:
                continue

            if "error" in utterance:
                message = utterance["message"]
                logger.error(f"Gladia error: {message}")
            elif "confidence" in utterance:
                type = utterance["type"]
                confidence = utterance["confidence"]
                transcript = utterance["transcription"]
                if confidence >= self._confidence:
                    if type == "final":
                        await self.push_frame(
                            TranscriptionFrame(transcript, "", time_now_iso8601())
                        )
                    else:
                        await self.push_frame(
                            InterimTranscriptionFrame(transcript, "", time_now_iso8601())
                        )
