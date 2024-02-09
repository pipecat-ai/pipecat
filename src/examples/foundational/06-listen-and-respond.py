import aiohttp
import asyncio
import os
from typing import AsyncGenerator

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.queue_aggregators import LLMAssistantContextAggregator, LLMContextAggregator, LLMUserContextAggregator
from examples.foundational.support.runner import configure
from dailyai.queue_frame import LLMMessagesQueueFrame, TranscriptionQueueFrame, QueueFrame, TextQueueFrame
from dailyai.services.ai_services import FrameLogger, AIService

class TranscriptFilter(AIService):
    def __init__(self, bot_participant_id=None):
        super().__init__()
        self.bot_participant_id = bot_participant_id
        print(f"Filtering transcripts from : {self.bot_participant_id}")

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if isinstance(frame, TranscriptionQueueFrame):
            if frame.participantId != self.bot_participant_id:
                yield frame

async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        global transport
        global llm
        global tts

        transport = DailyTransportService(
            room_url,
            token,
            "Respond bot",
            5,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=False
        )

        # llm = AzureLLMService(api_key=os.getenv("AZURE_CHATGPT_API_KEY"), endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"), model=os.getenv("AZURE_CHATGPT_MODEL"))
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_CHATGPT_API_KEY"))
        # tts = AzureTTSService(api_key=os.getenv("AZURE_SPEECH_API_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))
        tts = ElevenLabsTTSService(aiohttp_session=session, api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id="EXAVITQu4vr4xnSDxMaL")

        messages = [
            {"role": "system", "content": """You are Valerie, an agent for a company called Valorant Health. Your job is to help users get access to health care. You're talking to Chad Bailey, a 40 year old male who needs to see a doctor.

You need to do three things, in this order:

1. Confirm the user's identity.
2. Find out what kinds of doctors the user needs to see.
3. Get the name of their insurance company.

Start by introducing yourself and asking the user to verify their identity by providing their date of birth. Once their identity is confirmed, move on to step 2, then to step 3.

Once you have collected all of that information, respond with a JSON object containing the answers."""}
        ]
        tma_in = LLMUserContextAggregator(messages, transport._my_participant_id)
        tma_out = LLMAssistantContextAggregator(messages, transport._my_participant_id)
        # checklist = ChecklistProcessor(messages, llm)

        async def handle_transcriptions():
            tf = TranscriptFilter(transport._my_participant_id)
            await tts.run_to_queue(
                transport.send_queue,
                tma_out.run(
                    llm.run(
                        tma_in.run(
                            tf.run(
                                transport.get_receive_frames()
                            )
                        )         
                    )
                )
            )
        
        
        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            fl = FrameLogger("first other participant")
            await tts.run_to_queue(
                transport.send_queue,
                fl.run(
                    tma_out.run(
                        llm.run([LLMMessagesQueueFrame(messages)]),
                    )
                )            
            )
        
        transport.transcription_settings["extra"]["punctuate"] = True
        await asyncio.gather(transport.run(), handle_transcriptions())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
