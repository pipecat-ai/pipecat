import asyncio
import aiohttp
import logging
import os

import tkinter as tk

from dailyai.pipeline.frames import TextFrame
from dailyai.services.fal_ai_services import FalImageGenService
from dailyai.services.local_transport_service import LocalTransportService

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)

local_joined = False
participant_joined = False


async def main():
    async with aiohttp.ClientSession() as session:
        meeting_duration_minutes = 2
        tk_root = tk.Tk()
        tk_root.title("Calendar")
        transport = LocalTransportService(
            tk_root=tk_root,
            mic_enabled=True,
            camera_enabled=True,
            camera_width=1024,
            camera_height=1024,
            duration_minutes=meeting_duration_minutes,
        )

        imagegen = FalImageGenService(
            image_size="1024x1024",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"),
        )
        image_task = asyncio.create_task(
            imagegen.run_to_queue(
                transport.send_queue, [TextFrame("a cat in the style of picasso")]
            )
        )

        async def run_tk():
            while not transport._stop_threads.is_set():
                tk_root.update()
                tk_root.update_idletasks()
                await asyncio.sleep(0.1)

        await asyncio.gather(transport.run(), image_task, run_tk())


if __name__ == "__main__":
    asyncio.run(main())
