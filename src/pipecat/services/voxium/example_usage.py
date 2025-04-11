# main.py (Modified Example Usage)

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer # Keep VAD for input chunking
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
# Import the new service instead of the local Whisper service
from pipecat.services.voxium.stt import VoxiumSTTService # <--- IMPORT THE NEW SERVICE
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.transcriptions.language import Language # Optional for language setting
# Load .env file if present (contains VOXIUM_SERVER_URL, VOXIUM_API_KEY)
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG") # Use DEBUG to see connection logs etc.


class TranscriptionLogger(FrameProcessor):
    """Simple processor to print transcriptions."""
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # logger.DEBUG(f"TRANSCRIPTION: [{frame.language}] {frame.text}")
            logger.debug(f"TRANSCRIPTION: {frame.text}")


async def main():
    # Get server URL and API key from environment variables
    voxium_url = "wss://voxium.tech/asr/ws"
    voxium_api_key = os.getenv("VOXIUM_API_KEY")

    if not voxium_url:
        logger.error("VOXIUM_SERVER_URL environment variable not set.")
        sys.exit(1)
    if not voxium_api_key:
        logger.error("VOXIUM_API_KEY environment variable not set.")
        sys.exit(1)

    # Configure local audio transport
    # Keep VAD enabled here - it helps chunk the audio before sending to the service
    # The service itself doesn't *need* VAD, but sending smaller, VAD-segmented chunks
    # might align well with how the server expects to receive data.

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            # Keep input VAD - it helps segment the audio stream naturally
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True, # Pass audio even during silence if needed? Maybe False is better. Test this. Let's try True.
        )
    )

    # Instantiate the Voxium STT service
    stt = VoxiumSTTService(
        url=voxium_url,
        api_key=voxium_api_key,
        # Optional parameters (defaults shown):
        # language=Language.EN, # e.g., Language.ES, Language.FR, or None for auto
        # sample_rate=16000,   
        # vad_threshold=0.5,
        # silence_threshold_s=0.5,
        # speech_pad_ms=100,
        beam_size=3,
    )

    # Simple transcription logger
    tl = TranscriptionLogger()

    # Build the pipeline
    pipeline = Pipeline([
        transport.input(),
         # Get audio frames from local mic/vad
        stt,               # Send audio to Voxium server, receive transcriptions
        tl                 # Log the transcriptions
    ])

    # Create and run the pipeline task
    task = PipelineTask(pipeline)
    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    logger.info("Starting pipeline...")
    await runner.run(task)
    logger.info("Pipeline finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")