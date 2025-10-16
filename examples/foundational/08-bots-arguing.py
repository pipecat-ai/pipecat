import asyncio
import logging
import os
from typing import Tuple

import aiohttp
from dotenv import load_dotenv

from pipecat.frames.frames import AudioFrame, EndFrame, ImageFrame, LLMContextFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators import SentenceAggregator
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.daily import configure
from pipecat.services.azure import AzureLLMService, AzureTTSService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.fal import FalImageGenService
from pipecat.transports.daily.transport import DailyTransport

load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("pipecat")
logger.setLevel(logging.DEBUG)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
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
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="jBpfuIE2acCO8z3wKNLl",
        )
        dalle = FalImageGenService(
            params=FalImageGenService.InputParams(image_size="1024x1024"),
            aiohttp_session=session,
            key=os.getenv("FAL_KEY"),
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

        async def get_text_and_audio(messages) -> Tuple[str, bytearray]:
            """This function streams text from the LLM and uses the TTS service to convert
            that text to speech as it's received.
            """
            source_queue = asyncio.Queue()
            sink_queue = asyncio.Queue()
            sentence_aggregator = SentenceAggregator()
            pipeline = Pipeline([llm, sentence_aggregator, tts1], source_queue, sink_queue)

            await source_queue.put(LLMContextFrame(LLMContext(messages)))
            await source_queue.put(EndFrame())
            await pipeline.run_pipeline()

            message = ""
            all_audio = bytearray()
            while sink_queue.qsize():
                frame = sink_queue.get_nowait()
                if isinstance(frame, TextFrame):
                    message += frame.text
                elif isinstance(frame, AudioFrame):
                    all_audio.extend(frame.audio)

            return (message, all_audio)

        async def get_bot1_statement():
            message, audio = await get_text_and_audio(bot1_messages)

            bot1_messages.append({"role": "assistant", "content": message})
            bot2_messages.append({"role": "user", "content": message})

            return audio

        async def get_bot2_statement():
            message, audio = await get_text_and_audio(bot2_messages)

            bot2_messages.append({"role": "assistant", "content": message})
            bot1_messages.append({"role": "user", "content": message})

            return audio

        async def argue():
            for i in range(100):
                print(f"In iteration {i}")

                bot1_description = "A woman conservatively dressed as a librarian in a library surrounded by books, cartoon, serious, highly detailed"

                (audio1, image_data1) = await asyncio.gather(
                    get_bot1_statement(), dalle.run_image_gen(bot1_description)
                )
                await transport.send_queue.put(
                    [
                        ImageFrame(image_data1[1], image_data1[2]),
                        AudioFrame(audio1),
                    ]
                )

                bot2_description = "A cat dressed in a hot dog costume, cartoon, bright colors, funny, highly detailed"

                (audio2, image_data2) = await asyncio.gather(
                    get_bot2_statement(), dalle.run_image_gen(bot2_description)
                )
                await transport.send_queue.put(
                    [
                        ImageFrame(image_data2[1], image_data2[2]),
                        AudioFrame(audio2),
                    ]
                )

        await asyncio.gather(transport.run(), argue())


if __name__ == "__main__":
    asyncio.run(main())
