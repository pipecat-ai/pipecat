import asyncio
import aiohttp
import os
from dailyai.conversation_wrappers import InterruptibleConversationWrapper

from dailyai.queue_frame import StartStreamQueueFrame, TextQueueFrame
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.ai_services import FrameLogger

from examples.foundational.support.runner import configure


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
            context=context, api_key=os.getenv("OPENAI_CHATGPT_API_KEY"))

        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"))
        # tts = ElevenLabsTTSService(
        #     aiohttp_session=session,
        #     api_key=os.getenv("ELEVENLABS_API_KEY"),
        #     voice_id=os.getenv("ELEVENLABS_VOICE_ID"))
        fl = FrameLogger("just outside the innermost layer")

        async def run_response(in_frame, tma_in, tma_out):
            await tts.run_to_queue(
                transport.send_queue,
                # tma_out.run(
                llm.run(
                    # tma_in.run(
                    fl.run(
                        [StartStreamQueueFrame(), in_frame]
                    )
                    # )
                )
                # ),
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
