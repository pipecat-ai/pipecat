import asyncio
import aiohttp
import logging
import os
from dailyai.pipeline.frame_processor import FrameProcessor
from dailyai.pipeline.frames import TextFrame, TranscriptionFrame
from dailyai.pipeline.pipeline import Pipeline
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.transports.websocket_transport import WebsocketTransport
from dailyai.services.whisper_ai_services import WhisperSTTService

logging.basicConfig(format="%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)


class WhisperTranscriber(FrameProcessor):
    async def process_frame(self, frame):
        if isinstance(frame, TranscriptionFrame):
            print(f"Transcribed: {frame.text}")
        else:
            yield frame


async def main():
    async with aiohttp.ClientSession() as session:
        transport = WebsocketTransport(
            mic_enabled=True,
            speaker_enabled=True,
        )
        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        pipeline = Pipeline([
            WhisperSTTService(),
            WhisperTranscriber(),
            tts,
        ])

        @transport.on_connection
        async def queue_frame():
            await pipeline.queue_frames([TextFrame("Hello there!")])

        await transport.run(pipeline)

if __name__ == "__main__":
    asyncio.run(main())
