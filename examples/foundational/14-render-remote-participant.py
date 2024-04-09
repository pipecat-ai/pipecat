import asyncio
import io
import logging

from typing import AsyncGenerator

from PIL import Image

from dailyai.pipeline.aggregators import FrameProcessor

from dailyai.pipeline.frames import ImageFrame, Frame, UserImageFrame
from dailyai.pipeline.pipeline import Pipeline
from dailyai.transports.daily_transport import DailyTransport

from runner import configure

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


class UserImageProcessor(FrameProcessor):

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        print(frame)
        if isinstance(frame, UserImageFrame):
            yield ImageFrame(frame.image, frame.size)
        else:
            yield frame


async def main(room_url):
    transport = DailyTransport(
        room_url,
        token,
        "Render participant video",
        camera_width=1280,
        camera_height=720,
        camera_enabled=True,
        video_rendering_enabled=True
    )

    @ transport.event_handler("on_first_other_participant_joined")
    async def on_first_other_participant_joined(transport, participant):
        transport.render_participant_video(participant["id"])

    pipeline = Pipeline([UserImageProcessor()])

    await asyncio.gather(transport.run(pipeline))

if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
