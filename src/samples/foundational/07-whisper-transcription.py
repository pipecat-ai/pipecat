import argparse
import asyncio

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.whisper_ai_services import WhisperSTTService


async def main(room_url: str):
    global transport
    global stt

    transport = DailyTransportService(
        room_url,
        None,
        "Transcription bot",
    )
    transport.mic_enabled = False
    transport.camera_enabled = False
    transport.speaker_enabled = True
    stt = WhisperSTTService()
    transcription_output_queue = asyncio.Queue()

    async def handle_transcription():
        print("`````````TRANSCRIPTION`````````")
        while True:
            item = await transcription_output_queue.get()
            print(item.text)

    async def handle_speaker():
        await stt.run_to_queue(
            transcription_output_queue,
            transport.get_receive_frames()
        )
    await asyncio.gather(transport.run(), handle_speaker(), handle_transcription())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )

    args, unknown = parser.parse_known_args()
    asyncio.run(main(args.url))
