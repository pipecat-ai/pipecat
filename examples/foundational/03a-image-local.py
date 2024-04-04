import asyncio
import aiohttp
import logging
import os

import tkinter as tk

from dailyai.pipeline.frames import TextFrame, EndFrame
from dailyai.pipeline.pipeline import Pipeline
from dailyai.services.fal_ai_services import FalImageGenService
from dailyai.transports.local_transport import LocalTransport

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


async def main():
    async with aiohttp.ClientSession() as session:
        meeting_duration_minutes = 2

        tk_root = tk.Tk()
        tk_root.title("dailyai")

        transport = LocalTransport(
            tk_root=tk_root,
            mic_enabled=False,
            camera_enabled=True,
            camera_width=1024,
            camera_height=1024,
            duration_minutes=meeting_duration_minutes,
        )

        imagegen = FalImageGenService(
            image_size="square_hd",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"),
        )

        pipeline = Pipeline([imagegen])
        await pipeline.queue_frames([TextFrame("a cat in the style of picasso")])

        async def run_tk():
            while not transport._stop_threads.is_set():
                tk_root.update()
                tk_root.update_idletasks()
                await asyncio.sleep(0.1)

        await asyncio.gather(transport.run(pipeline, override_pipeline_source_queue=False), run_tk())


if __name__ == "__main__":
    asyncio.run(main())
