import asyncio
import logging

from dailyai.pipeline.frames import EndFrame, TranscriptionFrame
from dailyai.transports.local_transport import LocalTransport
from dailyai.services.whisper_ai_services import WhisperSTTService
from dailyai.pipeline.pipeline import Pipeline

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


async def main():
    meeting_duration_minutes = 1

    transport = LocalTransport(
        mic_enabled=True,
        camera_enabled=False,
        speaker_enabled=True,
        duration_minutes=meeting_duration_minutes,
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
    asyncio.run(main())
