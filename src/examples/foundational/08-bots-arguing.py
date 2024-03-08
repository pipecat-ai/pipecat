import aiohttp
import asyncio
import logging
import os

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.fal_ai_services import FalImageGenService
from dailyai.pipeline.frames import AudioFrame, ImageFrame
from examples.support.runner import configure

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


async def main(room_url: str):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransportService(
            room_url,
            None,
            "Respond bot",
            duration_minutes=10,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=True,
            camera_width=1024,
            camera_height=1024,
        )

        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"),
        )
        tts1 = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
        )
        tts2 = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="jBpfuIE2acCO8z3wKNLl",
        )
        dalle = FalImageGenService(
            image_size="1024x1024",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"),
        )

        bot1_messages = [
            {
                "role": "system",
                "content": "You are a stern librarian. You strongly believe that a hot dog is a sandwich. Start by stating this fact in a few sentences, then be prepared to debate this with the user. You shouldn't ever compromise on the fundamental truth that a hot dog is a sandwich. Your responses should only be a few sentences long.",
            },
        ]
        bot2_messages = [
            {
                "role": "system",
                "content": "You are a silly cat, and you strongly believe that a hot dog is not a sandwich. Debate this with the user, only responding with a few sentences. Don't ever accept that a hot dog is a sandwich.",
            },
        ]

        async def get_bot1_statement():
            # Run the LLMs synchronously for the back-and-forth
            bot1_msg = await llm.run_llm(bot1_messages)
            print(f"bot1_msg: {bot1_msg}")
            if bot1_msg:
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
            if bot2_msg:
                bot2_messages.append({"role": "assistant", "content": bot2_msg})
                bot1_messages.append({"role": "user", "content": bot2_msg})

            all_audio = bytearray()
            async for audio in tts2.run_tts(bot2_msg):
                all_audio.extend(audio)

            return all_audio

        async def argue():
            for i in range(100):
                print(f"In iteration {i}")

                bot1_description = "A woman conservatively dressed as a librarian in a library surrounded by books, cartoon, serious, highly detailed"

                (audio1, image_data1) = await asyncio.gather(
                    get_bot1_statement(), dalle.run_image_gen(bot1_description)
                )
                await transport.send_queue.put(
                    [
                        ImageFrame(None, image_data1[1]),
                        AudioFrame(audio1),
                    ]
                )

                bot2_description = "A cat dressed in a hot dog costume, cartoon, bright colors, funny, highly detailed"

                (audio2, image_data2) = await asyncio.gather(
                    get_bot2_statement(), dalle.run_image_gen(bot2_description)
                )
                await transport.send_queue.put(
                    [
                        ImageFrame(None, image_data2[1]),
                        AudioFrame(audio2),
                    ]
                )

        await asyncio.gather(transport.run(), argue())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
