import argparse
import asyncio
import re

from dailyai.services.ai_services import SentenceAggregator
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.queue_frame import QueueFrame, FrameType
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService

async def main(room_url:str):
    global transport
    global llm
    global tts

    transport = DailyTransportService(
        room_url,
        None,
        "Say Two Things Bot",
        1,
    )
    transport.mic_enabled = True
    transport.mic_sample_rate = 16000
    transport.camera_enabled = False

    llm = AzureLLMService()
    azure_tts = AzureTTSService()
    elevenlabs_tts = ElevenLabsTTSService(voice_id="ErXwobaYiN019PkySvjV")

    messages = [{"role": "system", "content": "tell the user a joke about llamas"}]

    # Start a task to run the LLM to create a joke, and convert the LLM output to audio frames. This task
    # will run in parallel with generating and speaking the audio for static text, so there's no delay to
    # speak the LLM response.
    buffer_queue = asyncio.Queue()
    llm_response_task = asyncio.create_task(
        elevenlabs_tts.run_to_queue(
            buffer_queue,
            SentenceAggregator().run(
                llm.run([QueueFrame(FrameType.LLM_MESSAGE, messages)])
            ),
            True,
        )
    )

    @transport.event_handler("on_participant_joined")
    async def on_joined(transport, participant):
        if participant["id"] == transport.my_participant_id:
            return

        await azure_tts.run_to_queue(
            transport.send_queue,
            [QueueFrame(FrameType.SENTENCE, "My friend the LLM is now going to tell a joke about llamas.")]
        )

        async def buffer_to_send_queue():
            while True:
                frame = await buffer_queue.get()
                await transport.send_queue.put(frame)
                buffer_queue.task_done()
                if frame.frame_type == FrameType.END_STREAM:
                    break

        await asyncio.gather(llm_response_task, buffer_to_send_queue())

        transport.stop_when_done()

    await transport.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )

    args, unknown = parser.parse_known_args()

    asyncio.run(main(args.url))
