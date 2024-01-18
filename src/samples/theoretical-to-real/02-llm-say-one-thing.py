import argparse
import asyncio
from typing import AsyncGenerator

from dailyai.queue_frame import QueueFrame, FrameType
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.ai_services import SentenceAggregator
from dailyai.services.azure_ai_services import AzureLLMService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService

async def main(room_url):
    meeting_duration_minutes = 1
    transport = DailyTransportService(
        room_url,
        None,
        "Say One Thing From an LLM",
        meeting_duration_minutes,
    )
    transport.mic_enabled = True

    tts = ElevenLabsTTSService(voice_id="29vD33N1CtxCmqQRPOHJ")
    llm = AzureLLMService()

    messages = [{
        "role": "system",
        "content": "You are an LLM in a WebRTC session, and this is a 'hello world' demo. Say hello to the world."
    }]
    tts_task = asyncio.create_task(
        tts.run_to_queue(
            transport.send_queue,
            SentenceAggregator().run(
                llm.run([QueueFrame(FrameType.LLM_MESSAGE, messages)])
            )
        )
    )

    @transport.event_handler("on_first_other_participant_joined")
    async def on_first_other_participant_joined(transport):
        await tts_task
        await transport.stop_when_done()

    await transport.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )

    args, unknown = parser.parse_known_args()
    asyncio.run(main(args.url))
