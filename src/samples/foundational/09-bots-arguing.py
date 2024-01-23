import argparse
import asyncio
import requests
import time
import urllib.parse

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.queue_frame import QueueFrame, FrameType

async def main(room_url:str):
    global transport
    global llm
    global tts

    transport = DailyTransportService(
        room_url,
        None,
        "Respond bot",
        5,
    )
    transport.mic_enabled = True
    transport.mic_sample_rate = 16000
    transport.camera_enabled = False

    llm = AzureLLMService()
    tts1 = AzureTTSService()
    tts2 = ElevenLabsTTSService()

    async def argue():
        bot1_messages = [
            {"role": "system", "content": "You strongly believe that a hot dog is a sandwich. Start by stating this fact in a few sentences, then be prepared to debate this with the user. Your responses should only be a few sentences long."},
        ]
        bot2_messages = [
            {"role": "system", "content": "You strongly believe that a hot dog is not a sandwich. Debate this with the user, only responding with a few sentences."},
        ]

        for i in range(1, 5):
            print(f"In iteration {i}")
            # Run the LLMs synchronously for the back-and-forth
            bot1_msg = await llm.run_llm(bot1_messages)
            print(f"bot1_msg: {bot1_msg}")
            bot1_messages.append({"role": "assistant", "content": bot1_msg})
            bot2_messages.append({"role": "user", "content": bot1_msg})

            await tts1.say(bot1_msg, transport.send_queue)

            bot2_msg = await llm.run_llm(bot2_messages)
            print(f"bot2_msg: {bot2_msg}")
            bot2_messages.append({"role": "assistant", "content": bot2_msg})
            bot1_messages.append({"role": "user", "content": bot2_msg})

            await tts2.say(bot2_msg, transport.send_queue)

    await asyncio.gather(transport.run(), argue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )

    args, unknown = parser.parse_known_args()

    asyncio.run(main(args.url))
