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

class ChecklistProcessor(AIService):
    def __init__(self, messages, llm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_step = 0
        self._messages = messages
        self._llm = llm
        self._id = "You are Valerie, an agent for a company called Valorant Health. Your job is to help users get access to health care. You're talking to Chad Bailey, a 40 year old male who needs to see a doctor."
        self._steps = [
            "Start by introducing yourself. Then, ask the user to confirm their identity by telling you their birthday. After the user has confirmed their identity, respond only with ABC.",
            "Now that the user has confirmed their identity, ask them to describe what kind of doctor they need to see. When the user has responded with at least one kind of doctor, respond only with ABC.",
            "Next, you need to ask the user what kind of health insurance they have. Once the user has told you what insurance company they use, respond only with ABC.",
            "Tell the user goodbye.",
            ""
        ]
        messages.append({"role": "system", "content": f"{self._id} {self._steps[0]}"})

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if isinstance(frame, TextQueueFrame):
            print(f"got a text frame: {frame.text}")
        if isinstance(frame, TextQueueFrame) and frame.text == "ABC":
            self._current_step += 1
            # yield TextQueueFrame(f"We should move on to Step {self._current_step}.")
            self._messages.append({"role": "system", "content": self._steps[self._current_step]})
            yield LLMMessagesQueueFrame(self._messages)
            print(f"past llmmessagesqueueframe yield")
            async for frame in llm.process_frame(LLMMessagesQueueFrame(self._messages)):
                yield frame
        else:
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
        )
        transport.mic_enabled = True
        transport.mic_sample_rate = 16000
        transport.camera_enabled = False

        # llm = AzureLLMService(api_key=os.getenv("AZURE_CHATGPT_API_KEY"), endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"), model=os.getenv("AZURE_CHATGPT_MODEL"))
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_CHATGPT_API_KEY"))
        # tts = AzureTTSService(api_key=os.getenv("AZURE_SPEECH_API_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))
        tts = ElevenLabsTTSService(aiohttp_session=session, api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id="EXAVITQu4vr4xnSDxMaL")

        messages = [
        ]
        tma_in = LLMUserContextAggregator(messages, transport._my_participant_id)
        tma_out = LLMAssistantContextAggregator(messages, transport._my_participant_id)
        checklist = ChecklistProcessor(messages, llm)

        async def handle_transcriptions():
            tf = TranscriptFilter(transport._my_participant_id)
            await tts.run_to_queue(
                transport.send_queue,
                checklist.run(
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
