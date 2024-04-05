import asyncio
import logging

from dailyai.pipeline.frames import EndFrame, TranscriptionFrame
from dailyai.transports.daily_transport import DailyTransport
from dailyai.services.whisper_ai_services import WhisperSTTService
from dailyai.pipeline.pipeline import Pipeline

from runner import configure

from dotenv import load_dotenv
load_dotenv(override=True)

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
    transport_done = asyncio.Event()

    pipeline = Pipeline([stt], source=transport.receive_queue, sink=transcription_output_queue)

    async def handle_transcription():
        print("`````````TRANSCRIPTION`````````")
        while not transport_done.is_set():
            item = await transcription_output_queue.get()
            print("got item from queue", item)
            if isinstance(item, TranscriptionFrame):
                print(item.text)
            elif isinstance(item, EndFrame):
                break
        print("handle_transcription done")

    async def run_until_done():
        await transport.run()
        transport_done.set()
        print("run_until_done done")

    await asyncio.gather(run_until_done(), pipeline.run_pipeline(), handle_transcription())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
