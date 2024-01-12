import argparse
import asyncio

from dailyai.queue_frame import QueueFrame, FrameType
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.open_ai_services import OpenAIImageGenService

local_joined = False
participant_joined = False

async def main(room_url):
    meeting_duration_minutes = 1
    transport = DailyTransportService(
        room_url,
        None,
        "Show a still frame image",
        meeting_duration_minutes,
    )
    transport.mic_enabled = False
    transport.camera_enabled = True
    transport.camera_width = 1024
    transport.camera_height = 1024

    imagegen = OpenAIImageGenService()
    image_task = asyncio.create_task(imagegen.run_image_gen("a cat in the style of picasso", "1024x1024"))

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        (_, image_bytes) = await image_task
        transport.output_queue.put(QueueFrame(FrameType.IMAGE_FRAME, image_bytes))

    await transport.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )

    args: argparse.Namespace = parser.parse_args()

    asyncio.run(main(args.url))
