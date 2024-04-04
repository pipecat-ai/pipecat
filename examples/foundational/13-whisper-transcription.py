import asyncio
import logging

from dailyai.transports.daily_transport import DailyTransport
from dailyai.services.whisper_ai_services import WhisperSTTService
from dailyai.pipeline.pipeline import Pipeline

from runner import configure

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


async def main(room_url: str):
    transport = DailyTransport(
        room_url,
        None,
        "Transcription bot",
        start_transcription=False,
        mic_enabled=False,
        camera_enabled=False,
        speaker_enabled=True,
    )

    stt = WhisperSTTService()

    transcription_output_queue = asyncio.Queue()

    pipeline = Pipeline([stt])
    pipeline.set_sink(transcription_output_queue)

    async def handle_transcription():
        print("`````````TRANSCRIPTION`````````")
        while True:
            item = await transcription_output_queue.get()
            print(item.text)

    await asyncio.gather(transport.run(pipeline), handle_transcription())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
