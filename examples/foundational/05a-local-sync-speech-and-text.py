import aiohttp
import asyncio
import logging
import tkinter as tk
import os
from dailyai.pipeline.aggregators import LLMFullResponseAggregator

from dailyai.pipeline.frames import AudioFrame, ImageFrame, LLMMessagesFrame, TextFrame
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.fal_ai_services import FalImageGenService
from dailyai.transports.local_transport import LocalTransport

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


async def main():
    async with aiohttp.ClientSession() as session:
        meeting_duration_minutes = 5
        tk_root = tk.Tk()
        tk_root.title("dailyai")

        transport = LocalTransport(
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
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo-preview")

        imagegen = FalImageGenService(
            image_size="1024x1024",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"),
        )

        # Get a complete audio chunk from the given text. Splitting this into its own
        # coroutine lets us ensure proper ordering of the audio chunks on the
        # send queue.
        async def get_all_audio(text):
            all_audio = bytearray()
            async for audio in tts.run_tts(text):
                all_audio.extend(audio)

            return all_audio

        async def get_month_description(aggregator, frame):
            async for frame in aggregator.process_frame(frame):
                if isinstance(frame, TextFrame):
                    return frame.text

        async def get_month_data(month):
            messages = [{"role": "system", "content": f"Describe a nature photograph suitable for use in a calendar, for the month of {
                month}. Include only the image description with no preamble. Limit the description to one sentence, please.", }]

            messages_frame = LLMMessagesFrame(messages)

            llm_full_response_aggregator = LLMFullResponseAggregator()

            image_description = None
            async for frame in llm.process_frame(messages_frame):
                result = await get_month_description(llm_full_response_aggregator, frame)
                if result:
                    image_description = result
                    break

            if not image_description:
                return

            to_speak = f"{month}: {image_description}"
            audio_task = asyncio.create_task(get_all_audio(to_speak))
            image_task = asyncio.create_task(
                imagegen.run_image_gen(image_description))
            (audio, image_data) = await asyncio.gather(audio_task, image_task)

            return {
                "month": month,
                "text": image_description,
                "image_url": image_data[0],
                "image": image_data[1],
                "audio": audio,
            }

        # We only specify 5 months as we create tasks all at once and we might
        # get rate limited otherwise.
        months: list[str] = [
            "January",
            "February",
            "March",
            "April",
            "May",
        ]

        async def show_images():
            # This will play the months in the order they're completed. The benefit
            # is we'll have as little delay as possible before the first month, and
            # likely no delay between months, but the months won't display in
            # order.
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

        month_tasks = [
            asyncio.create_task(
                get_month_data(month)) for month in months]

        await asyncio.gather(transport.run(), show_images(), run_tk())


if __name__ == "__main__":
    asyncio.run(main())
