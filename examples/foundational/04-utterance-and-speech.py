import asyncio
import logging
import os

import aiohttp
from dailyai.pipeline.merge_pipeline import SequentialMergePipeline
from dailyai.pipeline.pipeline import Pipeline

from dailyai.transports.daily_transport import DailyTransport
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.deepgram_ai_services import DeepgramTTSService
from dailyai.pipeline.frames import EndPipeFrame, LLMMessagesFrame, TextFrame
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService

from runner import configure

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


async def main(room_url: str):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            None,
            "Static And Dynamic Speech",
            duration_minutes=1,
            mic_enabled=True,
            mic_sample_rate=16000,
        )

        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"),
        )
        azure_tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
        )

        deepgram_tts = DeepgramTTSService(
            aiohttp_session=session,
            api_key=os.getenv("DEEPGRAM_API_KEY"),
        )
        elevenlabs_tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        messages = [{"role": "system",
                     "content": "tell the user a joke about llamas"}]

        # Start a task to run the LLM to create a joke, and convert the LLM output to audio frames. This task
        # will run in parallel with generating and speaking the audio for static text, so there's no delay to
        # speak the LLM response.
        llm_pipeline = Pipeline([llm, elevenlabs_tts])
        await llm_pipeline.queue_frames([LLMMessagesFrame(messages), EndPipeFrame()])

        simple_tts_pipeline = Pipeline([azure_tts])
        await simple_tts_pipeline.queue_frames(
            [
                TextFrame("My friend the LLM is going to tell a joke about llamas."),
                EndPipeFrame(),
            ]
        )

        merge_pipeline = SequentialMergePipeline(
            [simple_tts_pipeline, llm_pipeline])

        await asyncio.gather(
            transport.run(merge_pipeline),
            simple_tts_pipeline.run_pipeline(),
            llm_pipeline.run_pipeline(),
        )


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
