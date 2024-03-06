import asyncio
from re import S
import aiohttp
import os
from dailyai.pipeline.aggregators import GatedAggregator, LLMFullResponseAggregator, ParallelPipeline, SentenceAggregator

from dailyai.pipeline.frames import AudioFrame, EndFrame, ImageFrame, LLMMessagesQueueFrame, LLMResponseStartFrame
from dailyai.pipeline.pipeline import Pipeline
from dailyai.services.azure_ai_services import AzureLLMService, AzureImageGenServiceREST, AzureTTSService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.fal_ai_services import FalImageGenService
from dailyai.services.open_ai_services import OpenAIImageGenService

from examples.foundational.support.runner import configure


async def main(room_url):
    async with aiohttp.ClientSession() as session:
        meeting_duration_minutes = 5
        transport = DailyTransportService(
            room_url,
            None,
            "Month Narration Bot",
            duration_minutes=meeting_duration_minutes,
            mic_enabled=True,
            camera_enabled=True,
            mic_sample_rate=16000,
            camera_width=1024,
            camera_height=1024
        )

        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"))
        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="ErXwobaYiN019PkySvjV")

        dalle = FalImageGenService(
            image_size="1024x1024",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"))

        source_queue = asyncio.Queue()

        for month in ["January", "February"]:
            messages = [
                {
                    "role": "system",
                    "content": f"Describe a nature photograph suitable for use in a calendar, for the month of {month}. Include only the image description with no preamble. Limit the description to one sentence, please.",
                }
            ]
            await source_queue.put(LLMMessagesQueueFrame(messages))

        await source_queue.put(EndFrame())

        gated_aggregator = GatedAggregator(
            gate_open_fn=lambda frame: isinstance(frame, ImageFrame),
            gate_close_fn=lambda frame: isinstance(frame, LLMResponseStartFrame),
            start_open=False,
        )

        sentence_aggregator = SentenceAggregator()
        llm_full_response_aggregator = LLMFullResponseAggregator()

        pipeline = Pipeline(
            source=source_queue,
            sink=transport.send_queue,
            processors=[
                llm,
                sentence_aggregator,
                ParallelPipeline([[tts], [llm_full_response_aggregator, dalle]]),
                gated_aggregator,
            ],
        )
        pipeline_task = pipeline.run_pipeline()

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await pipeline_task

            # wait for the output queue to be empty, then leave the meeting
            await transport.stop_when_done()

        await transport.run()

if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
