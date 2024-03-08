import aiohttp
import argparse
import asyncio
import logging
import tkinter as tk
import os

from dailyai.pipeline.frames import AudioFrame, ImageFrame
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.fal_ai_services import FalImageGenService
from dailyai.services.local_transport_service import LocalTransportService

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


async def main(room_url):
    async with aiohttp.ClientSession() as session:
        meeting_duration_minutes = 5
        tk_root = tk.Tk()
        tk_root.title("Calendar")

        transport = LocalTransportService(
            mic_enabled=True,
            camera_enabled=True,
            camera_width=1024,
            camera_height=1024,
            duration_minutes=meeting_duration_minutes,
            tk_root=tk_root,
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_CHATGPT_API_KEY"), model="gpt-4-turbo-preview"
        )

        dalle = FalImageGenService(
            image_size="1024x1024",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"),
        )

        # Get a complete audio chunk from the given text. Splitting this into its own
        # coroutine lets us ensure proper ordering of the audio chunks on the send queue.
        async def get_all_audio(text):
            all_audio = bytearray()
            async for audio in tts.run_tts(text):
                all_audio.extend(audio)

            return all_audio

        async def get_month_data(month):
            messages = [
                {
                    "role": "system",
                    "content": f"Describe a nature photograph suitable for use in a calendar, for the month of {month}. Include only the image description with no preamble. Limit the description to one sentence, please.",
                }
            ]

            image_description = await llm.run_llm(messages)
            if not image_description:
                return

            to_speak = f"{month}: {image_description}"
            audio_task = asyncio.create_task(get_all_audio(to_speak))
            image_task = asyncio.create_task(dalle.run_image_gen(image_description))
            (audio, image_data) = await asyncio.gather(audio_task, image_task)

            return {
                "month": month,
                "text": image_description,
                "image_url": image_data[0],
                "image": image_data[1],
                "audio": audio,
            }

        months: list[str] = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        async def show_images():
            # This will play the months in the order they're completed. The benefit
            # is we'll have as little delay as possible before the first month, and
            # likely no delay between months, but the months won't display in order.
            for month_data_task in asyncio.as_completed(month_tasks):
                data = await month_data_task
                if data:
                    await transport.send_queue.put(
                        [
                            ImageFrame(data["image_url"], data["image"]),
                            AudioFrame(data["audio"]),
                        ]
                    )

            await asyncio.sleep(25)

            # wait for the output queue to be empty, then leave the meeting
            await transport.stop_when_done()

        async def run_tk():
            while not transport._stop_threads.is_set():
                tk_root.update()
                tk_root.update_idletasks()
                await asyncio.sleep(0.1)

        month_tasks = [asyncio.create_task(get_month_data(month)) for month in months]

        await asyncio.gather(transport.run(), show_images(), run_tk())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )

    args, unknown = parser.parse_known_args()

    asyncio.run(main(args.url))
