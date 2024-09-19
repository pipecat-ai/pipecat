

import json
from typing import AsyncGenerator
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame
)
from pipecat.services.ai_services import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from loguru import logger
from pipecat.services.tencent.common import credential
from pipecat.services.tencent.asr import speech_recognizer
import os
from dotenv import load_dotenv

load_dotenv()

APPID = os.getenv("TENCENT_APPID")
SECRET_ID = os.getenv("TENCENT_SECRET_ID")
SECRET_KEY = os.getenv("TENCENT_SECRET_KEY")
ENGINE_MODEL_TYPE = os.getenv("TENCENT_ENGINE_MODEL_TYPE", "16k_zh")


class TencentSTTService(STTService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._recognizer = None
        self._listener = None
        self._credential = credential.Credential(SECRET_ID, SECRET_KEY)

    async def set_model(self, model: str):
        logger.debug(f"Switching STT model to: [{model}]")
        global ENGINE_MODEL_TYPE
        ENGINE_MODEL_TYPE = model
        await self._reconnect()

    async def set_language(self, language: Language):
        logger.debug(f"Switching STT language to: [{language}]")
        await self._reconnect()

    async def start(self, frame: Frame):
        await super().start(frame)
        await self._initialize_recognizer()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self._recognizer:
            await self._recognizer.stop()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self._recognizer:
            await self._recognizer.stop()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:

        if self._recognizer.ws.open:
            await self._recognizer.write(audio)
        if not self._recognizer.ws.open:
            await self._reconnect()

        yield None

    async def _initialize_recognizer(self):
        # logger.debug("Initializing Tencent STT recognizer")
        self._listener = MySpeechRecognitionListener(self)
        self._recognizer = speech_recognizer.SpeechRecognizer(
            APPID, self._credential, ENGINE_MODEL_TYPE, self._listener
        )
        self._recognizer.set_filter_modal(1)
        self._recognizer.set_filter_punc(1)
        self._recognizer.set_filter_dirty(1)
        self._recognizer.set_need_vad(1)
        self._recognizer.set_voice_format(1)
        self._recognizer.set_word_info(1)
        self._recognizer.set_convert_num_mode(1)
        await self._recognizer.start()

    async def _on_message(self, response):
        result = response['result']
        if result['slice_type'] == 1:
            transcript = result['voice_text_str']
            await self.push_frame(InterimTranscriptionFrame(transcript, "", time_now_iso8601()))
        elif result['slice_type'] == 2:
            transcript = result['voice_text_str']
            logger.debug(f"Tencent STT: {transcript}")

            await self.push_frame(TranscriptionFrame(transcript, "", time_now_iso8601()))

    async def _reconnect(self):
        if self._recognizer:
            self._recognizer.stop()
        await self._initialize_recognizer()


class MySpeechRecognitionListener(speech_recognizer.SpeechRecognitionListener):
    def __init__(self, service):
        self.service = service
        self.partial_result = ""

    async def on_recognition_start(self, response):
        pass

    async def on_sentence_begin(self, response):
        pass

    async def on_recognition_result_change(self, response):
        await self.service._on_message(response)

    async def on_sentence_end(self, response):
        await self.service._on_message(response)

    async def on_fail(self, response):
        rsp_str = json.dumps(response, ensure_ascii=False)
        logger.error(f"Recognition failed: {rsp_str}")
