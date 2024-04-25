import asyncio
import aiohttp
import logging
import os
from pipecat.pipeline.frame_processor import FrameProcessor
from pipecat.pipeline.frames import TextFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.elevenlabs_ai_services import ElevenLabsTTSService
from pipecat.transports.websocket_transport import WebsocketTransport
from pipecat.services.whisper_ai_services import WhisperSTTService

logging.basicConfig(format="%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("pipecat")
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
