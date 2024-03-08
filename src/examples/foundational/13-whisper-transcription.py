import asyncio
import logging

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.whisper_ai_services import WhisperSTTService
from examples.support.runner import configure

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


async def main(room_url: str):
    transport = DailyTransportService(
        room_url,
        None,
        "Transcription bot",
        start_transcription=True,
        mic_enabled=False,
        camera_enabled=False,
        speaker_enabled=True,
    )

    stt = WhisperSTTService()
    transcription_output_queue = asyncio.Queue()

    async def handle_transcription():
        print("`````````TRANSCRIPTION`````````")
        while True:
            item = await transcription_output_queue.get()
            print(item.text)

    async def handle_speaker():
        await stt.run_to_queue(
            transcription_output_queue, transport.get_receive_frames()
        )

    await asyncio.gather(transport.run(), handle_speaker(), handle_transcription())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
