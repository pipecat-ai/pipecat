import argparse
import asyncio
import requests
import time
import urllib.parse

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.fal_ai_services import FalImageGenService
from dailyai.services.open_ai_services import OpenAIImageGenService
from dailyai.queue_aggregators import LLMContextAggregator
from dailyai.queue_frame import LLMMessagesQueueFrame, QueueFrame, TextQueueFrame
from dailyai.services.ai_services import AIService

from typing import AsyncGenerator, List

class TranscriptFilter(AIService):
    def __init__(self, bot_participant_id=None):
        self.bot_participant_id = bot_participant_id

    async def process_frame(self, frame:QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if frame.participantId != self.bot_participant_id:
            yield frame


async def main(room_url:str, token):
    global transport
    global llm
    global tts

    transport = DailyTransportService(
        room_url,
        token,
        "The Golden Kitty",
        5,
    )
    transport.mic_enabled = True
    transport.mic_sample_rate = 16000
    transport.camera_enabled = True
    transport.camera_width = 1024
    transport.camera_height = 1024

    llm = AzureLLMService()
    tts = ElevenLabsTTSService()

    @transport.event_handler("on_first_other_participant_joined")
    async def on_first_other_participant_joined(transport):
        await tts.say("Hi, I'm listening!", transport.send_queue)

    async def handle_transcriptions():
        messages = [
            {"role": "system", "content": "You are the Golden Kitty, the mascot for Product Hunt's annual awards. You are a cat who knows everything about all the cool new tech startups. You should be clever, and a bit sarcastic. You should also tell jokes every once in a while.  Your responses should only be a few sentences long."},
        ]

        tma_in = LLMContextAggregator(
            messages, "user", transport.my_participant_id
        )
        tma_out = LLMContextAggregator(
            messages, "assistant", transport.my_participant_id
        )
        tf = TranscriptFilter(transport.my_participant_id)
        await tts.run_to_queue(
            transport.send_queue,
            tma_out.run(
                llm.run(
                    tma_in.run(
                        tf.run(
                            transport.get_receive_frames()
                        )
                    )
                )
            )
        )

    async def make_cats():
        imagegen = OpenAIImageGenService(image_size="1024x1024")

        while True:
            print("generating new image")
            await imagegen.run_to_queue(transport.send_queue, [TextQueueFrame("a golden kitty trophy, cartoon, colorful, detailed, 4k")])
            await asyncio.sleep(10)
        
    transport.transcription_settings["extra"]["punctuate"] = True
    await asyncio.gather(transport.run(), handle_transcriptions(), make_cats())





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

    # Create a meeting token for the given room with an expiration 24 hours in the future.
    room_name: str = urllib.parse.urlparse(args.url).path[1:]
    expiration: float = time.time() + 60 * 60 * 24

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
