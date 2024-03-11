import asyncio
import aiohttp
import logging
import os

from dailyai.pipeline.aggregators import (
    LLMAssistantContextAggregator,
    LLMResponseAggregator,
    LLMUserContextAggregator,
    UserResponseAggregator,
)
from dailyai.pipeline.frames import ImageFrame, SpriteFrame
from dailyai.pipeline.pipeline import Pipeline
from dailyai.services.ai_services import FrameLogger
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from examples.support.runner import configure

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)

sprites = {}
image_files = [
    "sc-default.png",
    "sc-talk.png",
    "sc-listen-1.png",
    "sc-think-1.png",
    "sc-think-2.png",
    "sc-think-3.png",
    "sc-think-4.png",
]

script_dir = os.path.dirname(__file__)

for file in image_files:
    # Build the full path to the image file
    full_path = os.path.join(script_dir, "assets", file)
    # Get the filename without the extension to use as the dictionary key
    filename = os.path.splitext(os.path.basename(full_path))[0]
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites[file] = img.tobytes()

# When the bot isn't talking, show a static image of the cat listening
quiet_frame = ImageFrame("", sprites["sc-listen-1.png"])
# When the bot is talking, build an animation from two sprites
talking_list = [sprites["sc-default.png"], sprites["sc-talk.png"]]
talking = [random.choice(talking_list) for x in range(30)]
talking_frame = SpriteFrame(images=talking)

# TODO: Support "thinking" as soon as we get a valid transcript, while LLM is processing
thinking_list = [
    sprites["sc-think-1.png"],
    sprites["sc-think-2.png"],
    sprites["sc-think-3.png"],
    sprites["sc-think-4.png"],
]
thinking_frame = SpriteFrame(images=thinking_list)


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransportService(
            room_url,
            token,
            "Respond bot",
            duration_minutes=5,
            start_transcription=True,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=False,
            vad_enabled=True,
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_CHATGPT_API_KEY"), model="gpt-4-turbo-preview"
        )

        pipeline = Pipeline([FrameLogger(), llm, FrameLogger(), tts])

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await tts.say("Hi, I'm listening!", transport.send_queue)

        async def run_conversation():
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio. Respond to what the user said in a creative and helpful way.",
                },
            ]

            await transport.run_interruptible_pipeline(
                pipeline,
                post_processor=LLMResponseAggregator(messages),
                pre_processor=UserResponseAggregator(messages),
            )

        transport.transcription_settings["extra"]["punctuate"] = False
        await asyncio.gather(transport.run(), run_conversation())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
