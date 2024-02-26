import asyncio
import aiohttp
import logging
import os

from dailyai.conversation_wrappers import InterruptibleConversationWrapper

from dailyai.queue_frame import StartStreamQueueFrame, TextQueueFrame
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.deepgram_ai_services import DeepgramTTSService
from dailyai.services.ai_services import FrameLogger
from dailyai.services.groq_ai_services import GroqLLMService
from dailyai.queue_aggregators import LLMContextAggregator

from support.runner import configure


logging.basicConfig(format=f"%(asctime)s - %(levelname)s: %(message)s")  # or whatever
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)

async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        context = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio. Respond to what the user said in a creative and helpful way.",
            },
        ]
        transport = DailyTransportService(
            room_url,
            token,
            "Respond bot",
            duration_minutes=5,
            start_transcription=True,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=False,
            # TODO-CB: Should this be VAD enabled or something?
            speaker_enabled=True,
            context=context
        )

        # llm = AzureLLMService(
        #     api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
        #     endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
        #     model=os.getenv("AZURE_CHATGPT_MODEL"),
        #     context=context)
        llm = OpenAILLMService(
            context=context, api_key=os.getenv("OPENAI_CHATGPT_API_KEY"), model="gpt-3.5-turbo")
        llm = GroqLLMService(api_key=os.getenv("GROQ_API_KEY"), model="mixtral-8x7b-32768", context=context)
        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"))
        # tts = ElevenLabsTTSService(
        #     aiohttp_session=session,
        #     api_key=os.getenv("ELEVENLABS_API_KEY"),
        #     voice_id=os.getenv("ELEVENLABS_VOICE_ID"))
        # tts = DeepgramTTSService(aiohttp_session=session, api_key=os.getenv("DEEPGRAM_API_KEY"), voice=os.getenv("DEEPGRAM_VOICE"), split_sentences=True)
        fl = FrameLogger("just outside the innermost layer")
        lca = LLMContextAggregator(context=context, bot_participant_id=transport._my_participant_id)
        
        # TODO-CB: Is this just a super powerful callback?
        async def run_response(in_frame):
            await tts.run_to_queue(
                transport.send_queue,
                llm.run(
                    lca.run(
                        [StartStreamQueueFrame(), in_frame]
                    )
                )
            )

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await tts.say("Hi, I'm listening!", transport.send_queue)

        transport.transcription_settings["extra"]["endpointing"] = True
        transport.transcription_settings["extra"]["punctuate"] = True
        await asyncio.gather(transport.run(), transport.run_conversation(run_response))


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
