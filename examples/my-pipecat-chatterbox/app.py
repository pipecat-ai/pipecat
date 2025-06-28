import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from pipecat.pipeline.pipeline import Pipeline

# This import path will now work correctly after the installation
from pipecat.services.ollama.llm import OllamaLLM
from pipecat.transports.services.simple_transport import SimpleTransport
from pipecat.vad.silero import SileroVAD
from pipecat.services.deepgram import DeepgramSTTService

from chatterbox_service import ChatterboxTTSService

dotenv_path = Path(__file__).parent.parent / ".env"
print(f"Loading environment variables from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")


async def main():
    transport = SimpleTransport()

    if not DEEPGRAM_API_KEY:
        print("Error: DEEPGRAM_API_KEY not found in environment. Please check your .env file.")
        return

    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY, model="nova-2-general")
    llm = OllamaLLM(model="gemma3n")
    tts = ChatterboxTTSService()

    pipeline = Pipeline(
        [
            transport.input(),
            SileroVAD(),
            stt,
            llm,
            tts,
            transport.output(),
        ]
    )

    print("Pipecat server with local Chatterbox TTS is running...")
    await pipeline.run_async()


if __name__ == "__main__":
    asyncio.run(main())
