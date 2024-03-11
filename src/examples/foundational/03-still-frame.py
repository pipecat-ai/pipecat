import asyncio
import aiohttp
import logging
import os

from dailyai.pipeline.frames import TextFrame
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.fal_ai_services import FalImageGenService

from examples.support.runner import configure

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


async def main(room_url):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransportService(
            room_url,
            None,
            "Show a still frame image",
            camera_enabled=True,
            camera_width=1024,
            camera_height=1024,
            duration_minutes=1
        )

        imagegen = FalImageGenService(
            image_size="square_hd",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"),
        )

        other_joined_event = asyncio.Event()

        async def show_image():
            await other_joined_event.wait()
            await imagegen.run_to_queue(
                transport.send_queue, [TextFrame("a cat in the style of picasso")]
            )

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            other_joined_event.set()

        await asyncio.gather(transport.run(), show_image())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
