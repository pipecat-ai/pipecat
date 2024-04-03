import asyncio
import aiohttp
import logging
import os
from typing import AsyncGenerator

from dailyai.pipeline.aggregators import (
    SentenceAggregator,
)
from dailyai.pipeline.frames import (
    Frame,
    LLMMessagesFrame,
    TextFrame,
    SendAppMessageFrame,
)
from dailyai.pipeline.frame_processor import FrameProcessor
from dailyai.pipeline.pipeline import Pipeline
from dailyai.transports.daily_transport import DailyTransport
from dailyai.services.azure_ai_services import AzureTTSService
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.pipeline.aggregators import LLMFullResponseAggregator

from runner import configure

from dotenv import load_dotenv


load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)

"""
This example looks a bit different than the chatbot example, because it isn't waiting on the user to stop talking to start translating.
It also isn't saving what the user or bot says into the context object for use in subsequent interactions.
"""


# We need to use a custom service here to yield LLM frames without saving
# any context
class TranslationProcessor(FrameProcessor):
    def __init__(self, language):
        self._language = language

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, TextFrame):
            context = [
                {
                    "role": "system",
                    "content": f"You will be provided with a sentence in English, and your task is to translate it into {self._language}.",
                },
                {"role": "user", "content": frame.text},
            ]
            yield LLMMessagesFrame(context)
        else:
            yield frame


class TranslationSubtitles(FrameProcessor):
    def __init__(self, language):
        self._language = language

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, TextFrame):
            app_message = {
                "language": self._language,
                "text": frame.text
            }
            yield SendAppMessageFrame(app_message, None)
            yield frame
        else:
            yield frame


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Translator",
            duration_minutes=5,
            start_transcription=True,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=False,
        )
        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            voice="es-ES-AlvaroNeural",
        )
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4-turbo-preview"
        )
        sa = SentenceAggregator()
        tp = TranslationProcessor("Spanish")
        lfra = LLMFullResponseAggregator()
        ts = TranslationSubtitles("spanish")
        pipeline = Pipeline([sa, tp, llm, lfra, ts, tts])

        transport.transcription_settings["extra"]["endpointing"] = True
        transport.transcription_settings["extra"]["punctuate"] = True
        await transport.run(pipeline)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
