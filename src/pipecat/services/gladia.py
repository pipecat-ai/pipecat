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
            "language": self.language_to_service_language(params.language)
            if params.language
            else Language.EN,
            "transcription_hint": params.transcription_hint,
            "endpointing": params.endpointing,
            "prosody": params.prosody,
        }
        self._confidence = confidence

    def language_to_service_language(self, language: Language) -> str | None:
        match language:
            case Language.BG:
                return "bulgarian"
            case Language.CA:
                return "catalan"
            case Language.ZH:
                return "chinese"
            case Language.CS:
                return "czech"
            case Language.DA:
                return "danish"
            case Language.NL:
                return "dutch"
            case (
                Language.EN
                | Language.EN_US
                | Language.EN_AU
                | Language.EN_GB
                | Language.EN_NZ
                | Language.EN_IN
            ):
                return "english"
            case Language.ET:
                return "estonian"
            case Language.FI:
                return "finnish"
            case Language.FR | Language.FR_CA:
                return "french"
            case Language.DE | Language.DE_CH:
                return "german"
            case Language.EL:
                return "greek"
            case Language.HI:
                return "hindi"
            case Language.HU:
                return "hungarian"
            case Language.ID:
                return "indonesian"
            case Language.IT:
                return "italian"
            case Language.JA:
                return "japanese"
            case Language.KO:
                return "korean"
            case Language.LV:
                return "latvian"
            case Language.LT:
                return "lithuanian"
            case Language.MS:
                return "malay"
            case Language.NO:
                return "norwegian"
            case Language.PL:
                return "polish"
            case Language.PT | Language.PT_BR:
                return "portuguese"
            case Language.RO:
                return "romanian"
            case Language.RU:
                return "russian"
            case Language.SK:
                return "slovak"
            case Language.ES:
                return "spanish"
            case Language.SV:
                return "slovenian"
            case Language.TH:
                return "thai"
            case Language.TR:
                return "turkish"
            case Language.UK:
                return "ukrainian"
            case Language.VI:
                return "vietnamese"
        return None

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
