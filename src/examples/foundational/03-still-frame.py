import asyncio
import aiohttp
import os

from dailyai.pipeline.frames import TextQueueFrame
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.fal_ai_services import FalImageGenService
from dailyai.services.open_ai_services import OpenAIImageGenService
from dailyai.services.azure_ai_services import AzureImageGenServiceREST

from examples.foundational.support.runner import configure

local_joined = False
participant_joined = False


async def main(room_url):
    async with aiohttp.ClientSession() as session:
        meeting_duration_minutes = 1
        transport = DailyTransportService(
            room_url,
            None,
            "Show a still frame image",
            duration_minutes=meeting_duration_minutes,
            mic_enabled=False,
            camera_enabled=True,
            camera_width=1024,
            camera_height=1024
        )

        imagegen = FalImageGenService(
            image_size="1024x1024",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"))
        # imagegen = OpenAIImageGenService(aiohttp_session=session, api_key=os.getenv("OPENAI_DALLE_API_KEY"), image_size="1024x1024")
        # imagegen = AzureImageGenServiceREST(image_size="1024x1024", aiohttp_session=session, api_key=os.getenv("AZURE_DALLE_API_KEY"), endpoint=os.getenv("AZURE_DALLE_ENDPOINT"), model=os.getenv("AZURE_DALLE_MODEL"))

        image_task = asyncio.create_task(
            imagegen.run_to_queue(
                transport.send_queue, [
                    TextQueueFrame("a cat in the style of picasso")]))

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await image_task

        await transport.run()


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
