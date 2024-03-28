import asyncio
import aiohttp
import logging
import os
from dailyai.pipeline.frames import EndFrame, TextFrame
from dailyai.pipeline.pipeline import Pipeline

from dailyai.transports.daily_transport import DailyTransport
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService

from runner import configure

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


async def main(room_url):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            None,
            "Say One Thing",
            mic_enabled=True,
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        pipeline = Pipeline([tts])

        # Register an event handler so we can play the audio when the
        # participant joins.
        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            if participant["info"]["isLocal"]:
                return

            participant_name = participant["info"]["userName"] or ''
            await pipeline.queue_frames([TextFrame("Hello there, " + participant_name + "!"), EndFrame()])

        await transport.run(pipeline)
        del tts


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
