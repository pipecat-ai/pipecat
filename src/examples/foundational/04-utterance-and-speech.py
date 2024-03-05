import asyncio
import os

import aiohttp
from dailyai.pipeline.pipeline import Pipeline

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.pipeline.frames import EndStreamQueueFrame, LLMMessagesQueueFrame
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService

from examples.foundational.support.runner import configure


async def main(room_url: str):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransportService(
            room_url,
            None,
            "Static And Dynamic Speech",
            duration_minutes=1,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=False
        )

        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"))
        azure_tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"))
        elevenlabs_tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"))

        messages = [{"role": "system", "content": "tell the user a joke about llamas"}]

        # Start a task to run the LLM to create a joke, and convert the LLM output to audio frames. This task
        # will run in parallel with generating and speaking the audio for static text, so there's no delay to
        # speak the LLM response.
        buffer_queue = asyncio.Queue()
        source_queue = asyncio.Queue()
        pipeline = Pipeline(source = source_queue, sink=buffer_queue, processors=[llm, elevenlabs_tts])
        source_queue.put_nowait(LLMMessagesQueueFrame(messages))
        pipeline_run_task = pipeline.run_pipeline()

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await azure_tts.say("My friend the LLM is now going to tell a joke about llamas.", transport.send_queue)

            async def buffer_to_send_queue():
                while True:
                    frame = await buffer_queue.get()
                    await transport.send_queue.put(frame)
                    buffer_queue.task_done()
                    if isinstance(frame, EndStreamQueueFrame):
                        break

            await asyncio.gather(pipeline_run_task, buffer_to_send_queue())

            await transport.stop_when_done()

        await transport.run()


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
