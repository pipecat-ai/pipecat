import aiohttp
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
from dailyai.queue_frame import QueueFrame, AudioQueueFrame, ImageQueueFrame

async def main(room_url:str):
    async with aiohttp.ClientSession() as session:
        global transport
        global llm
        global tts

        transport = DailyTransportService(
            room_url,
            None,
            "Respond bot",
            600,
        )
        transport.mic_enabled = True
        transport.mic_sample_rate = 16000
        transport.camera_enabled = True
        transport.camera_width = 1024
        transport.camera_height = 1024

        llm = AzureLLMService()
        tts1 = AzureTTSService()
        tts2 = ElevenLabsTTSService(session)
        dalle = FalImageGenService(image_size="1024x1024", aiohttp_session=session)
        # dalle = OpenAIImageGenService(image_size="1024x1024")

        topic = "Are pokemon edible?"
        affirmative = "A woman dressed as a cowboy, outside on a ranch"
        negative = "Pikachu in a business suit"
        
        topic = "Is a hot dog a sandwich?"
        affirmative = "A woman conservatively dressed as a librarian in a library surrounded by books"
        negative = "A cat dressed in a hot dog costume"



        bot1_messages = [
            {"role": "system", "content": f"You are {affirmative}. You're in a debate, and the topic is: '{topic}'. You're arguing the affirmative. Start by stating this fact in a few sentences, then be prepared to debate this with the user. You shouldn't ever agree with the user. Your responses should only be a few sentences long."},
        ]
        bot2_messages = [
            {"role": "system", "content": f"You are {negative}. You're in a debate, and the topic is: '{topic}'. You're arguing the negative.  Debate this with the user, only responding with a few sentences. Don't ever agree with the user."},
        ]
        
        async def get_bot1_statement():
            # Run the LLMs synchronously for the back-and-forth
            bot1_msg = await llm.run_llm(bot1_messages)
            print(f"bot1_msg: {bot1_msg}")
            bot1_messages.append({"role": "assistant", "content": bot1_msg})
            bot2_messages.append({"role": "user", "content": bot1_msg})

            all_audio = bytearray()
            async for audio in tts1.run_tts(bot1_msg):
                all_audio.extend(audio)

            return all_audio
        
        async def get_bot2_statement():
            # Run the LLMs synchronously for the back-and-forth
            bot2_msg = await llm.run_llm(bot2_messages)
            print(f"bot2_msg: {bot2_msg}")
            bot2_messages.append({"role": "assistant", "content": bot2_msg})
            bot1_messages.append({"role": "user", "content": bot2_msg})

            all_audio = bytearray()
            async for audio in tts2.run_tts(bot2_msg):
                all_audio.extend(audio)

            return all_audio

        async def argue():
            for i in range(100):
                print(f"In iteration {i}")

                bot1_description = f"{affirmative}, cartoon, highly detailed"

                (audio1, image_data1) = await asyncio.gather(
                    get_bot1_statement(), dalle.run_image_gen(bot1_description)
                )
                await transport.send_queue.put(
                    [
                        ImageQueueFrame(None, image_data1[1]),
                        AudioQueueFrame(audio1),
                    ]
                )

                bot2_description = f"{negative}, cartoon, bright colors, funny, highly detailed"

                (audio2, image_data2) = await asyncio.gather(
                    get_bot2_statement(), dalle.run_image_gen(bot2_description)
                )
                await transport.send_queue.put(
                    [
                        ImageQueueFrame(None, image_data2[1]),
                        AudioQueueFrame(audio2),
                    ]
                )

        await asyncio.gather(transport.run(), argue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )

    args, unknown = parser.parse_known_args()

    asyncio.run(main(args.url))
