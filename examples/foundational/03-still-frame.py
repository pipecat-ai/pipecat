import asyncio
import aiohttp
import logging
import os

from dailyai.pipeline.frames import TextFrame
from dailyai.pipeline.pipeline import Pipeline
from dailyai.transports.daily_transport import DailyTransport
from dailyai.services.fal_ai_services import FalImageGenService

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

        pipeline = Pipeline([imagegen])

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            # Note that we do not put an EndFrame() item in the pipeline for this demo.
            # This means that the bot will stay in the channel until it times out.
            # An EndFrame() in the pipeline would cause the transport to shut
            # down.
            await pipeline.queue_frames(
                [TextFrame("a cat in the style of picasso")]
            )

        await transport.run(pipeline)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
