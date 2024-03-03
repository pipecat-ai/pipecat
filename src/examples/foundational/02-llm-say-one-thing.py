import asyncio
import os

import aiohttp

from dailyai.pipeline.frames import LLMMessagesQueueFrame
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.deepgram_ai_services import DeepgramTTSService
from dailyai.services.open_ai_services import OpenAILLMService
from examples.foundational.support.runner import configure


async def main(room_url):
    async with aiohttp.ClientSession() as session:
        meeting_duration_minutes = 1
        transport = DailyTransportService(
            room_url,
            None,
            "Say One Thing From an LLM",
            duration_minutes=meeting_duration_minutes,
            mic_enabled=True
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"))
        # tts = AzureTTSService(api_key=os.getenv("AZURE_SPEECH_API_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))
        # tts = DeepgramTTSService(aiohttp_session=session, api_key=os.getenv("DEEPGRAM_API_KEY"), voice=os.getenv("DEEPGRAM_VOICE"))

        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"))
        # llm = OpenAILLMService(api_key=os.getenv("OPENAI_CHATGPT_API_KEY"))
        messages = [{
            "role": "system",
            "content": "You are an LLM in a WebRTC session, and this is a 'hello world' demo. Say hello to the world."
        }]
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
