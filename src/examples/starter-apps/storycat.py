import aiohttp
import asyncio
import json
import random
import logging
import os
import re
import wave
from typing import AsyncGenerator
from PIL import Image

from dailyai.pipeline.pipeline import Pipeline
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.open_ai_services import OpenAILLMService
from dailyai.services.deepgram_ai_services import DeepgramTTSService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService
from dailyai.pipeline.aggregators import (
    LLMAssistantContextAggregator,
    LLMContextAggregator,
    LLMUserContextAggregator,
    UserResponseAggregator,
    LLMResponseAggregator,
)
from examples.support.runner import configure
from dailyai.pipeline.frames import (
    LLMMessagesQueueFrame,
    TranscriptionQueueFrame,
    Frame,
    TextFrame,
    LLMFunctionCallFrame,
    LLMFunctionStartFrame,
    LLMResponseEndFrame,
    StartFrame,
    AudioFrame,
    SpriteFrame,
    ImageFrame,
)
from dailyai.services.ai_services import FrameLogger, AIService

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("dailyai")
logger.setLevel(logging.DEBUG)

sounds = {}
sound_files = ["clack-short.wav", "clack.wav", "clack-short-quiet.wav"]

script_dir = os.path.dirname(__file__)

for file in sound_files:
    # Build the full path to the image file
    full_path = os.path.join(script_dir, "assets", file)
    # Get the filename without the extension to use as the dictionary key
    filename = os.path.splitext(os.path.basename(full_path))[0]
    # Open the image and convert it to bytes
    with wave.open(full_path) as audio_file:
        sounds[file] = audio_file.readframes(-1)


steps = [
    {
        "prompt": "Start by introducing yourself. Then, ask the user to confirm their identity by telling you their birthday, including the year. When they answer with their birthday, call the verify_birthday function.",
        "run_async": False,
        "failed": "The user provided an incorrect birthday. Ask them for their birthday again. When they answer, call the verify_birthday function.",
        "tools": [
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
                                "description": "The user's birthdate, including the year. The user can provide it in any format, but convert it to YYYY-MM-DD format to call this function.",
                            }
                        },
                    },
                },
            }
        ],
    },
    {
        "prompt": "Next, thank the user for confirming their identity, then ask the user to list their current prescriptions. Each prescription needs to have a medication name and a dosage. Do not call the list_prescriptions function with any unknown dosages.",
        "run_async": True,
        "tools": [
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
                                        "medication": {
                                            "type": "string",
                                            "description": "The medication's name",
                                        },
                                        "dosage": {
                                            "type": "string",
                                            "description": "The prescription's dosage",
                                        },
                                    },
                                },
                            }
                        },
                    },
                },
            }
        ],
    },
    {
        "prompt": "Next, ask the user if they have any allergies. Once they have listed their allergies or confirmed they don't have any, call the list_allergies function.",
        "run_async": True,
        "tools": [
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
                                            "description": "What the user is allergic to",
                                        }
                                    },
                                },
                            }
                        },
                    },
                },
            }
        ],
    },
    {
        "prompt": "Now ask the user if they have any medical conditions the doctor should know about. Once they've answered the question, call the list_conditions function.",
        "run_async": True,
        "tools": [
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
                                            "description": "The user's medical condition",
                                        }
                                    },
                                },
                            }
                        },
                    },
                },
            },
        ],
    },
    {
        "prompt": "Finally, ask the user the reason for their doctor visit today. Once they answer, call the list_visit_reasons function.",
        "run_async": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "list_visit_reasons",
                    "description": "Once the user has provided a list of the reasons they are visiting a doctor today, call this function.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "visit_reasons": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "The user's reason for visiting the doctor",
                                        }
                                    },
                                },
                            }
                        },
                    },
                },
            }
        ],
    },
    {
        "prompt": "Now, thank the user and end the conversation.",
        "run_async": True,
        "tools": [],
    },
    {"prompt": "", "run_async": True, "tools": []},
]
current_step = 0


class TranscriptFilter(AIService):
    def __init__(self, bot_participant_id=None):
        super().__init__()
        self.bot_participant_id = bot_participant_id

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, TranscriptionQueueFrame):
            if frame.participantId != self.bot_participant_id:
                yield frame


class ChecklistProcessor(AIService):
    def __init__(self, messages, llm, tools, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._messages = messages
        self._llm = llm
        self._tools = tools
        self._id = "You are Jessica, an agent for a company called Tri-County Health Services. Your job is to collect important information from the user before their doctor visit. You're talking to Chad Bailey. You should address the user by their first name and be polite and professional. You're not a medical professional, so you shouldn't provide any advice. Keep your responses short. Your job is to collect information to give to a doctor. Don't make assumptions about what values to plug into functions. Ask for clarification if a user response is ambiguous."
        self._acks = ["One sec.", "Let me confirm that.", "Thanks.", "OK."]

        # Create an allowlist of functions that the LLM can call
        self._functions = [
            "verify_birthday",
            "list_prescriptions",
            "list_allergies",
            "list_conditions",
            "list_visit_reasons",
        ]

        messages.append(
            {"role": "system", "content": f"{self._id} {steps[0]['prompt']}"}
        )

    def verify_birthday(self, args):
        return args["birthday"] == "1983-01-01"

    def list_prescriptions(self, args):
        # print(f"--- Prescriptions: {args['prescriptions']}\n")
        pass

    def list_allergies(self, args):
        # print(f"--- Allergies: {args['allergies']}\n")
        pass

    def list_conditions(self, args):
        # print(f"--- Medical Conditions: {args['conditions']}")
        pass

    def list_visit_reasons(self, args):
        # print(f"Visit Reasons: {args['visit_reasons']}")
        pass

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        global current_step
        this_step = steps[current_step]
        # TODO-CB: forcing a global here :/
        self._tools.clear()
        self._tools.extend(this_step["tools"])
        if isinstance(frame, LLMFunctionStartFrame):
            print(f"... Preparing function call: {frame.function_name}")
            self._function_name = frame.function_name
            if this_step["run_async"]:
                # Get the LLM talking about the next step before getting the rest
                # of the function call completion
                current_step += 1
                self._messages.append(
                    {"role": "system", "content": steps[current_step]["prompt"]}
                )
                yield LLMMessagesQueueFrame(self._messages)
                async for frame in llm.process_frame(
                    LLMMessagesQueueFrame(self._messages), tool_choice="none"
                ):
                    yield frame
            else:
                # Insert a quick response while we run the function
                # yield AudioFrame(sounds["clack-short-quiet.wav"])
                pass
        elif isinstance(frame, LLMFunctionCallFrame):

            if frame.function_name and frame.arguments:
                print(f"--> Calling function: {frame.function_name} with arguments:")
                pretty_json = re.sub(
                    "\n", "\n    ", json.dumps(json.loads(frame.arguments), indent=2)
                )
                print(f"--> {pretty_json}\n")
                if not frame.function_name in self._functions:
                    raise Exception(
                        f"The LLM tried to call a function named {frame.function_name}, which isn't in the list of known functions. Please check your prompt and/or self._functions."
                    )
                fn = getattr(self, frame.function_name)
                result = fn(json.loads(frame.arguments))

                if not this_step["run_async"]:
                    if result:
                        current_step += 1
                        self._messages.append(
                            {"role": "system", "content": steps[current_step]["prompt"]}
                        )
                        yield LLMMessagesQueueFrame(self._messages)
                        async for frame in llm.process_frame(
                            LLMMessagesQueueFrame(self._messages), tool_choice="none"
                        ):
                            yield frame
                    else:
                        self._messages.append(
                            {"role": "system", "content": this_step["failed"]}
                        )
                        yield LLMMessagesQueueFrame(self._messages)
                        async for frame in llm.process_frame(
                            LLMMessagesQueueFrame(self._messages), tool_choice="none"
                        ):
                            yield frame
                    print(f"<-- Verify result: {result}\n")

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
            "Story Cat",
            5,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=False,
            start_transcription=True,
            vad_enabled=True,
        )
        # TODO-CB: Go back to vad_enabled

        messages = []
        tools = []

        # llm = AzureLLMService(api_key=os.getenv("AZURE_CHATGPT_API_KEY"), endpoint=os.getenv(
        #     "AZURE_CHATGPT_ENDPOINT"), model=os.getenv("AZURE_CHATGPT_MODEL"))
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_CHATGPT_API_KEY"),
            model="gpt-4-1106-preview",
            tools=tools,
        )  # gpt-4-1106-preview
        # tts = AzureTTSService(api_key=os.getenv(
        #     "AZURE_SPEECH_API_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))
        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="XrExE9yKIg1WjnnlVkGX",
        )  # matilda
        # tts = DeepgramTTSService(aiohttp_session=session, api_key=os.getenv(
        #     "DEEPGRAM_API_KEY"), voice="aura-asteria-en")

        # lca = LLMContextAggregator(
        #     messages=messages, bot_participant_id=transport._my_participant_id)
        checklist = ChecklistProcessor(messages, llm, tools)
        fl = FrameLogger("FRAME LOGGER 1:")
        fl2 = FrameLogger("FRAME LOGGER 2:")

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            fl = FrameLogger("first other participant")
            # TODO-CB: Make sure this message gets into the context somehow
            await tts.run_to_queue(
                transport.send_queue,
                llm.run([LLMMessagesQueueFrame(messages)]),
            )

        async def handle_intake():
            pipeline = Pipeline(processors=[fl, llm, fl2, checklist, tts])
            await transport.run_interruptible_pipeline(
                pipeline,
                post_processor=LLMResponseAggregator(messages),
                pre_processor=UserResponseAggregator(messages),
            )

        transport.transcription_settings["extra"]["endpointing"] = True
        transport.transcription_settings["extra"]["punctuate"] = True
        try:
            await asyncio.gather(transport.run(), handle_intake())
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("whoops")
            transport.stop()


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
