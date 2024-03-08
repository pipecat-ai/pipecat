import asyncio
import os
import logging

import aiohttp

from dailyai.pipeline.frames import LLMMessagesQueueFrame
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.deepgram_ai_services import DeepgramTTSService
from dailyai.services.open_ai_services import OpenAILLMService
from examples.support.runner import configure

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)

async def main(room_url):
    async with aiohttp.ClientSession() as session:
        meeting_duration_minutes = 1
        transport = DailyTransportService(
            room_url,
            None,
            "Say One Thing From an LLM",
            duration_minutes=meeting_duration_minutes,
            mic_enabled=True,
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_CHATGPT_API_KEY"), model="gpt-4-turbo-preview"
        )
        messages = [
            {
                "role": "system",
                "content": "You are an LLM in a WebRTC session, and this is a 'hello world' demo. Say hello to the world.",
            }
        ]
        tts_task = asyncio.create_task(
            tts.run_to_queue(
                transport.send_queue,
                llm.run([LLMMessagesQueueFrame(messages)]),
            )
        )

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await tts_task
            await transport.stop_when_done()

        await transport.run()


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
