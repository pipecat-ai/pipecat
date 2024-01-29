import argparse
import asyncio
import os
from typing import AsyncGenerator
import aiohttp
import requests
import time
import urllib.parse

from PIL import Image
from dailyai.queue_frame import ImageQueueFrame, QueueFrame

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.ai_services import AIService
from dailyai.queue_aggregators import LLMAssistantContextAggregator, LLMUserContextAggregator
from dailyai.services.fal_ai_services import FalImageGenService


class ImageSyncAggregator(AIService):
    def __init__(self, speaking_path: str, waiting_path: str):
        self._speaking_image = Image.open(speaking_path)
        self._speaking_image_bytes = self._speaking_image.tobytes()

        self._waiting_image = Image.open(waiting_path)
        self._waiting_image_bytes = self._waiting_image.tobytes()

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        yield ImageQueueFrame(None, self._speaking_image_bytes)
        yield frame
        yield ImageQueueFrame(None, self._waiting_image_bytes)


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as aiohttp_session:
        transport = DailyTransportService(
            room_url,
            token,
            "Respond bot",
            5,
        )
        transport.camera_enabled = True
        transport.camera_width = 1024
        transport.camera_height = 1024
        transport.mic_enabled = True
        transport.mic_sample_rate = 16000

        llm = AzureLLMService()
        tts = AzureTTSService()
        img = FalImageGenService(image_size="1024x1024", aiohttp_session=aiohttp_session)

        async def get_images():
            get_speaking_task = asyncio.create_task(
                img.run_image_gen("An image of a cat speaking")
            )
            get_waiting_task = asyncio.create_task(
                img.run_image_gen("An image of a cat waiting")
            )

            (speaking_data, waiting_data) = await asyncio.gather(
                get_speaking_task, get_waiting_task
            )

            return speaking_data, waiting_data

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await tts.say("Hi, I'm listening!", transport.send_queue)

        async def handle_transcriptions():
            messages = [
                {"role": "system", "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio. Respond to what the user said in a creative and helpful way."},
            ]

            tma_in = LLMUserContextAggregator(
                messages, transport.my_participant_id
            )
            tma_out = LLMAssistantContextAggregator(
                messages, transport.my_participant_id
            )
            image_sync_aggregator = ImageSyncAggregator(
                os.path.join(os.path.dirname(__file__), "images", "speaking.png"),
                os.path.join(os.path.dirname(__file__), "images", "waiting.png"),
            )
            await tts.run_to_queue(
                transport.send_queue,
                image_sync_aggregator.run(
                    tma_out.run(
                        llm.run(
                            tma_in.run(
                                transport.get_receive_frames()
                            )
                        )
                    )
                )
            )

        transport.transcription_settings["extra"]["punctuate"] = True
        await asyncio.gather(transport.run(), handle_transcriptions())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )
    parser.add_argument(
        "-k",
        "--apikey",
        type=str,
        required=True,
        help="Daily API Key (needed to create token)",
    )

    args, unknown = parser.parse_known_args()

    # Create a meeting token for the given room with an expiration 1 hour in the future.
    room_name: str = urllib.parse.urlparse(args.url).path[1:]
    expiration: float = time.time() + 60 * 60

    res: requests.Response = requests.post(
        f"https://api.daily.co/v1/meeting-tokens",
        headers={"Authorization": f"Bearer {args.apikey}"},
        json={
            "properties": {"room_name": room_name, "is_owner": True, "exp": expiration}
        },
    )

    if res.status_code != 200:
        raise Exception(f"Failed to create meeting token: {res.status_code} {res.text}")

    token: str = res.json()["token"]

    asyncio.run(main(args.url, token))
