import asyncio
import logging
import tkinter as tk

from typing import AsyncGenerator

from dailyai.pipeline.aggregators import FrameProcessor

from dailyai.pipeline.frames import ImageFrame, Frame, UserImageFrame
from dailyai.pipeline.pipeline import Pipeline
from dailyai.transports.daily_transport import DailyTransport

from dailyai.transports.local_transport import LocalTransport
from runner import configure

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


class UserImageProcessor(FrameProcessor):

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, UserImageFrame):
            yield ImageFrame(frame.image, frame.size)
        else:
            yield frame


async def main(room_url: str, token):
    tk_root = tk.Tk()
    tk_root.title("dailyai")

    local_transport = LocalTransport(
        tk_root=tk_root,
        camera_enabled=True,
        camera_width=1280,
        camera_height=720
    )

    transport = DailyTransport(
        room_url,
        token,
        "Render participant video",
        video_rendering_enabled=True
    )

    @transport.event_handler("on_first_other_participant_joined")
    async def on_first_other_participant_joined(transport, participant):
        transport.render_participant_video(participant["id"])

    async def run_tk():
        while not transport._stop_threads.is_set():
            tk_root.update()
            tk_root.update_idletasks()
            await asyncio.sleep(0.1)

    local_pipeline = Pipeline([UserImageProcessor()], source=transport.receive_queue)

    await asyncio.gather(
        transport.run(),
        local_transport.run(local_pipeline, override_pipeline_source_queue=False),
        run_tk()
    )

if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
