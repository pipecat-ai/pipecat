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
from dailyai.queue_frame import LLMMessagesQueueFrame, TranscriptionQueueFrame, QueueFrame, TextQueueFrame, LLMFunctionCallFrame
from dailyai.services.ai_services import FrameLogger, AIService

tools = [
    {
        "type": "function",
        "function": {
            "name": "verify_birthday",
            "description": "Use this function to verify the user has provided their correct birthday.",
            "parameters": {
                "type": "object",
                "properties": {
                    "birthday": {
                        "type": "string",
                        "description": "The user's birthdate. Convert it to YYYY-MM-DD format."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_prescriptions",
            "description": "Once the user has provided a list of their prescription medications, call this function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prescriptions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The medication's name"
                                },
                                "dosage": {
                                    "type": "string",
                                    "description": "The prescription's dosage"
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_allergies",
            "description": "Once the user has provided a list of their allergies, call this function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "allergies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "What the user is allergic to"
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_conditions",
            "description": "Once the user has provided a list of their medical conditions, call this function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The user's medical condition"
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_visit_reasons",
            "description": "Once the user has provided a list of the reasons they are visiting a doctor today, call this function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasons": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The user's reason for visiting the doctor"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
]


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
        self._id = "You are Jessica, an agent for a company called Tri-County Advanced Optimum Health Solution Specialists. Your job is to collect important information from the user before they visit a doctor. You're talking to Chad Bailey. You should address the user by their first name and be polite and professional. You're not a medical professional, so you shouldn't provide any advice. Keep your responses short. Your job is to collect information to give to a doctor."
        self._steps = [
            "Start by introducing yourself. Then, ask the user to confirm their identity by telling you their birthday. When they answer with their birthday, call the verify_birthday function.",
            "You've already confirmed the user's birthday, so don't call the verify_birthday function. Ask the user to list their current prescriptions. If the user responds with one or two prescriptions, ask them to confirm it's the complete list. Make sure each medication also includes the dosage. Once the user has provided all their prescriptions, call the list_prescriptions function.",
            "Don't call the verify_birthday or list_prescription functions. Ask the user if they have any allergies. Once they have listed their allergies or confirmed they don't have any, call the list_allergies function.",
            "Don't call the verify_birthday, list_allergies, or list_prescriptions functions. Ask the user if they have any medical conditions the doctor should know about. Once they've answered the question, call the list_conditions function."
            "Ask the user the reason for their doctor visit today. Once they answer, double-check to make sure they don't have any other health concerns. After that, call the list_visit_reasons function.",
            "Now, thank the user and end the conversation.",
            ""
        ]
        messages.append(
            {"role": "system", "content": f"{self._id} {self._steps[0]}"})

    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        if isinstance(frame, LLMFunctionCallFrame):
            print(f"FUNCTION CALL: {frame}")
            self._current_step += 1
            # yield TextQueueFrame(f"We should move on to Step {self._current_step}.")
            self._messages.append({
                "role": "system", "content": f"{self._id} {self._steps[self._current_step]}"})
            yield LLMMessagesQueueFrame(self._messages)
            async for frame in llm.process_frame(LLMMessagesQueueFrame(self._messages), tool_choice="none"):
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
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=False,
            start_transcription=True
        )

        # llm = AzureLLMService(api_key=os.getenv("AZURE_CHATGPT_API_KEY"), endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"), model=os.getenv("AZURE_CHATGPT_MODEL"))
        llm = OpenAILLMService(api_key=os.getenv(
            "OPENAI_CHATGPT_API_KEY"), model="gpt-4", tools=tools)
        # tts = AzureTTSService(api_key=os.getenv(
        #     "AZURE_SPEECH_API_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))
        tts = ElevenLabsTTSService(aiohttp_session=session, api_key=os.getenv(
            "ELEVENLABS_API_KEY"), voice_id="EXAVITQu4vr4xnSDxMaL")
        messages = [
        ]
        tma_in = LLMUserContextAggregator(
            messages, transport._my_participant_id)
        tma_out = LLMAssistantContextAggregator(
            messages, transport._my_participant_id)
        checklist = ChecklistProcessor(messages, llm)
        fl = FrameLogger("got transcript")

        async def handle_transcriptions():
            tf = TranscriptFilter(transport._my_participant_id)
            await tts.run_to_queue(
                transport.send_queue,
                checklist.run(
                    tma_out.run(
                        llm.run(
                            tma_in.run(
                                tf.run(
                                    fl.run(
                                        transport.get_receive_frames()
                                    )
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
        try:
            await asyncio.gather(transport.run(), handle_transcriptions())
        except (asyncio.CancelledError, KeyboardInterrupt):
            transport.stop()


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
