import asyncio
import aiohttp
import logging
import os

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService

from examples.support.runner import configure

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


async def main(room_url):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransportService(
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

        other_joined_event = asyncio.Event()
        participant_name = ''

        async def say_hello():
            nonlocal tts
            nonlocal participant_name

            await other_joined_event.wait()
            print("done waiting")
            await tts.say(
                "Hello there, " + participant_name + "!",
                transport.send_queue,
            )
            await transport.stop_when_done()

        # Register an event handler so we can play the audio when the participant joins.
        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            if participant["info"]["isLocal"]:
                return

            nonlocal participant_name
            participant_name = participant["info"]["userName"] or ''
            other_joined_event.set()

        await asyncio.gather(transport.run(), say_hello())
        del tts


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
