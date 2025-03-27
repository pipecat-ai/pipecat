import asyncio
import os
from dotenv import load_dotenv

from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAIService
from pipecat.transports.services.daily import DailyParams, DailyTransport

# Load environment variables
load_dotenv()

async def main():
    # Initialize Daily for phone call handling
    transport = DailyTransport(
        room_url=os.getenv("DAILY_ROOM_URL"),
        token="",  # leave empty, token is not your API key
        bot_name="AI Assistant",
        params=DailyParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            start_video_off=True,
            start_audio_off=True
        )
    )

    # Initialize speech-to-text with Deepgram
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY")
    )

    # Initialize text-to-speech with Cartesia
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID")
    )

    # Initialize OpenAI for conversation
    llm = OpenAIService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4-turbo-preview"
    )

    # Create the pipeline
    pipeline = Pipeline([
        stt,  # Convert speech to text
        llm,  # Process with OpenAI
        tts,  # Convert response to speech
        transport.output()  # Output to phone call
    ])

    # Create pipeline runner
    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    # Handle incoming audio from phone call
    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        participant_name = participant.get("info", {}).get("userName", "there")
        await task.queue_frame(TextFrame(f"Hello {participant_name}! I'm your AI assistant. How can I help you today?"))

    # Handle when participant leaves
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.cancel()

    # Run the pipeline
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main()) 