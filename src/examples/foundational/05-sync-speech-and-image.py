import asyncio
from typing import Any, AsyncGenerator, Callable, Tuple
import aiohttp
import os
from dailyai.queue_aggregators import QueueFrameAggregator, QueueMergeGateOnFirst, QueueTee

from dailyai.queue_frame import AudioQueueFrame, EndStreamQueueFrame, ImageQueueFrame, LLMMessagesQueueFrame, LLMResponseEndQueueFrame, QueueFrame, TextQueueFrame
from dailyai.services.ai_services import PipeService
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

        """

                                      / TTS                   \
        Month prompt -> LLM -> Fork ->                         -> Gate -> Transport
                                      \ Aggregate -> ImageGen /
        """

        month_description_queue: asyncio.Queue[QueueFrame] = asyncio.Queue()
        llm = AzureLLMService(
            source_queue=month_description_queue,
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"),
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="ErXwobaYiN019PkySvjV")

        dalle = FalImageGenService(
            image_size="1024x1024",
            aiohttp_session=session,
            key_id=os.getenv("FAL_KEY_ID"),
            key_secret=os.getenv("FAL_KEY_SECRET"),
        )

        def aggregator(
            accumulation, frame: QueueFrame
        ) -> tuple[Any, QueueFrame | None]:
            if not accumulation:
                accumulation = ""

            if isinstance(frame, TextQueueFrame):
                accumulation += frame.text
                return (accumulation, None)
            elif isinstance(frame, LLMResponseEndQueueFrame):
                return ("", TextQueueFrame(accumulation))
            else:
                return (accumulation, frame)

        # This queue service takes chunks from LLM output and merges them into one text frame
        # that will be used to prompt the image service.
        llm_aggregator_for_image = QueueFrameAggregator(aggregator=aggregator, finalizer=lambda x: None)

        # Set the source queue for the image service to the sink of the aggregator service
        dalle.source_queue = llm_aggregator_for_image.sink_queue

        # This queue service takes the output from the LLM and sends it to the TTS service and
        # the aggregator for the image generation service.
        tee = QueueTee(source_queue=llm.sink_queue, sinks=[tts, llm_aggregator_for_image])

        # This queue service takes input from the TTS service and the image service, and waits
        # to forward any audio frames until the image generation is complete. It will send
        # the image first, then the audio frames; this ensures that the image is shown before
        # the audio associated with the image is played.
        tts_image_gate = QueueMergeGateOnFirst([dalle.sink_queue, tts.sink_queue])

        # We send the image of this queue service to the transport output.
        tts_image_gate.sink_queue = transport.send_queue

        # Queue up all the months in the LLM service source queue
        months = ["January"] #, "February"]
        for month in months:
            messages = [
                {
                    "role": "system",
                    "content": f"Describe a nature photograph suitable for use in a calendar, for the month of {month}. Include only the image description with no preamble. Limit the description to one sentence, please.",
                }
            ]

            await month_description_queue.put(LLMMessagesQueueFrame(messages))

        await month_description_queue.put(EndStreamQueueFrame())

        await asyncio.gather(transport.run(), *[service.process_queue() for service in [llm, tts, dalle, tee, tts_image_gate, llm_aggregator_for_image]])

if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
