import argparse
import asyncio
from typing import AsyncGenerator

from dailyai.queue_frame import QueueFrame, FrameType
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService

local_joined = False
participant_joined = False

async def main(room_url):
    meeting_duration_minutes = 1
    transport = DailyTransportService(
        room_url,
        None,
        "Say One Thing From an LLM",
        meeting_duration_minutes,
    )
    transport.mic_enabled = True

    text_to_llm_queue = asyncio.Queue()
    llm_to_tts_queue = asyncio.Queue()

    tts = ElevenLabsTTSService(
        llm_to_tts_queue, transport.get_async_output_queue(), voice_id="29vD33N1CtxCmqQRPOHJ"
    )
    llm = AzureLLMService(text_to_llm_queue, llm_to_tts_queue)

    messages = [{
        "role": "system",
        "content": "You are an LLM in a WebRTC session, and this is a 'hello world' demo. Say hello to the world."
    }]
    await text_to_llm_queue.put(QueueFrame(FrameType.LLM_MESSAGE_FRAME, messages))
    await text_to_llm_queue.put(QueueFrame(FrameType.END_STREAM, None))

    llm_task = asyncio.create_task(llm.run())

    has_joined = False
    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        nonlocal has_joined
        if participant["id"] == transport.my_participant_id or has_joined:
            return

        has_joined = True
        await asyncio.gather(llm_task, tts.run())

        # wait for the output queue to be empty, then leave the meeting
        transport.output_queue.join()
        transport.stop()

    await transport.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )

    args, unknown = parser.parse_known_args()
    asyncio.run(main(args.url))
