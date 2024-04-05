import asyncio
import aiohttp
import logging
import os

from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.transports.local_transport import LocalTransport

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


async def main():
    async with aiohttp.ClientSession() as session:
        meeting_duration_minutes = 1
        transport = LocalTransport(
            duration_minutes=meeting_duration_minutes, mic_enabled=True
        )
        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        async def say_something():
            await asyncio.sleep(1)
            await transport.say("Hello there.", tts)
            await transport.stop_when_done()

        await asyncio.gather(transport.run(), say_something())


if __name__ == "__main__":
    asyncio.run(main())
