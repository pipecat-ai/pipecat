import asyncio

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.whisper_ai_services import WhisperSTTService

from samples.foundational.support.runner import configure

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
    (url, token) = configure()
    asyncio.run(main(url))