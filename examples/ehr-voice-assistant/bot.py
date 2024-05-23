import asyncio
import aiohttp
import copy
import json
import os
import re
import sys
import wave
from typing import List
from PIL import Image

from openai._types import NotGiven, NOT_GIVEN

from openai.types.chat import (
    ChatCompletionToolParam,
)

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator, LLMAssistantContextAggregator
from pipecat.processors.logger import FrameLogger
from pipecat.frames.frames import (
    Frame,
    LLMMessagesFrame,
    LLMFunctionCallFrame,
    LLMFunctionStartFrame,
    AudioRawFrame,
    ImageRawFrame,
    SpriteFrame,
    TTSStoppedFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.services.ai_services import AIService
from pipecat.transports.services.daily import DailyParams, DailyTranscriptionSettings, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.services.openai import OpenAILLMContext, OpenAILLMContextFrame

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sprites = []

script_dir = os.path.dirname(__file__)

for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(ImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

flipped = sprites[::-1]
sprites.extend(flipped)

# When the bot isn't talking, show a static image of the cat listening
quiet_frame = sprites[0]
talking_frame = SpriteFrame(images=sprites)


class TalkingAnimation(FrameProcessor):
    """
    This class starts a talking animation when it receives an first AudioFrame,
    and then returns to a "quiet" sprite when it sees a TTSStoppedFrame.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, AudioRawFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        elif isinstance(frame, TTSStoppedFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame)


def execute_function_call(frame):
    if frame.function_name == "get_schedule":
        args = json.loads(frame.arguments)
        results = get_schedule(args)
        return results
    elif frame.function_name == "get_clinical_record":
        args = json.loads(frame.arguments)
        results = get_clinical_record(args)
        return results

    else:
        results = f"Error: function {frame.function_name} does not exist"

    return results


def get_schedule(args):
    schedule = [
        {"appointment_id": 1,
         "patient_name": "Janet Lee",
         "patient_id": 0,
         "appointment_start": "8:30am",
         "appointment_end": "9:00am",
         "reason_for_visit": "Experiencing shortness of breath",
         "last_visit": "June 30, 2023"},
        {"appointment_id": 2,
         "patient_name": "Carl Freeman",
         "patient_id": 1,
         "appointment_start": "9:00am",
         "appointment_end": "9:30am",
         "reason_for_visit": "Sprained ankle",
         "last_visit": "August 29, 2023"},
        {"appointment_id": 3,
         "patient_name": "Steve Dobson",
         "patient_id": 2,
         "appointment_start": "9:30am",
         "appointment_end": "10:00am",
         "reason_for_visit": "Flu-like symptoms",
         "last_visit": "March 23, 2023"},
        {"appointment_id": 4,
         "patient_name": "Steve Dobson",
         "patient_id": 3,
         "appointment_start": "10:00am",
         "appointment_end": "10:30am",
         "reason_for_visit": "Flu-like symptoms",
         "last_visit": "March 23, 2023"},
        {"appointment_id": 5,
         "patient_name": "Steve Dobson",
         "patient_id": 4,
         "appointment_start": "10:30am",
         "appointment_end": "11:00am",
         "reason_for_visit": "Flu-like symptoms",
         "last_visit": "March 23, 2023"},
        {"appointment_id": 6,
         "patient_name": "Steve Dobson",
         "patient_id": 5,
         "appointment_start": "11:00am",
         "appointment_end": "11:30am",
         "reason_for_visit": "Flu-like symptoms",
         "last_visit": "March 23, 2023"},
        {"appointment_id": 7,
         "patient_name": "Steve Dobson",
         "patient_id": 6,
         "appointment_start": "12:30pm",
         "appointment_end": "1:00pm",
         "reason_for_visit": "Flu-like symptoms",
         "last_visit": "March 23, 2023"},
        {"appointment_id": 8,
         "patient_name": "Steve Dobson",
         "patient_id": 7,
         "appointment_start": "1:00pm",
         "appointment_end": "3:00pm",
         "reason_for_visit": "Flu-like symptoms",
         "last_visit": "March 23, 2023"},
        {"appointment_id": 9,
         "patient_name": "Steve Dobson",
         "patient_id": 8,
         "appointment_start": "3:00pm",
         "appointment_end": "5:30pm",
         "reason_for_visit": "Flu-like symptoms",
         "last_visit": "March 23, 2023"},
    ]
    return schedule


def get_clinical_record(args):
    patient_id = int(args['patient_id'])
    details = [
        {"allergies": "none",
         "medical_history": "Heart disease, emphysema, smoking",
            "prescriptions": [{"name": "Lisinopril", "dosage": "20mg", "frequency": "daily"}],
            "vaccinations_current": True,
         },
        {"allergies": "latex, penicillin",
         "medical_history": "Obesity",
         "prescriptions": [{"name": "Gabapentin", "dosage": "200mg", "frequency": "daily"}],
         "vaccinations_current": False,
         }
    ]
    return details[patient_id]


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_schedule",
            "description": "Get a physician's schedule for the day",
            "parameters": {
                "type": "object",
                "properties": {
                    "physician_name": {
                        "type": "string",
                        "description": "The name of the physician whose schedule you're trying to fetch",
                    },
                },
                "required": ["physician_name"],
            },
        }},
    {
        "type": "function",
        "function": {
                "name": "get_clinical_record",
                "description": "Returns a summary of the patient's clinical history including allergies, vitals, prescription medications, immunizations, and so forth.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The ID of the patient"
                        }

                    }
                }

        }
    }
]

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant for physicians. I'm Dr. Bailey. I'm very busy, and you've worked with me for a long time, so you can skip the pleasantries and keep your responses short and to the point. You will not search the web unless explicitly requested. Don't provide extra information that I don't ask for directly. Emulate clear writers like Steinbeck, Hemingway, Orwell. Use British English. Use a sincere voice like David Foster Wallace or Kurt Vonnegut. Emulate great comedians like George Carlin. Favor short, clear sentences. Stay organized; be proactive. Treat me as an expert in all fields. Be accurate; mistakes erode my trust. Offer uncommon recommendations. Avoid the word “not”. Value reason over authority. Encourage contrarian ideas. Allow speculation; flag when used. Limit lectures on safety and morality. Be succinct. No introductions. No conclusions. Respect content policies; explain when needed. Keep a neutral tone, but be opinionated. Be specific, not abstract. Use rich language without prefaces or summaries. Never use cliches or platitudes. If I ask about my schedule for the day, just tell me about my first appointment. Begin by telling me 'good morning, Dr. Bailey', and asking how you can help me.",
    },
]


class FunctionCallProcessor(FrameProcessor):
    def __init__(
        self,
        context: OpenAILLMContext,
        llm: AIService,
        *args,
        **kwargs,
    ):
        self._context = context
        self._llm = llm
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, LLMFunctionStartFrame):
            # play a sound effect, maybe
            pass
        elif isinstance(frame, LLMFunctionCallFrame):
            print(f"!!! time to call a function, {frame}")
            # call the function, put the response in the context, and reprompt
            result = execute_function_call(frame)
            print(f"!!! result is {result}")
            print(f"!!! frame {frame}")
            print(f"!!! frame tool call id {frame.tool_call_id}")
            print(f"!!! frame function name {frame.function_name}")

            self._context.add_message({"role": "function",
                                       "tool_call_id": frame.tool_call_id,
                                       "name": frame.function_name,
                                       "content": json.dumps(result)})
            print(f"!!! function call done, context messages is now {self._context.get_messages()}")
            print(f"### starting the replacement llm yield...")
            await self.push_frame(OpenAILLMContextFrame(self._context), FrameDirection.UPSTREAM)
            print(f"### done with the replacement push")
        else:
            await self.push_frame(frame)


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=576,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
            )
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="EXAVITQu4vr4xnSDxMaL",
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o")

        context = OpenAILLMContext(
            messages=messages,
            tools=tools,
        )

        fcp = FunctionCallProcessor(context=context, llm=llm)
        fl1 = FrameLogger("^^^ Before LUCA")
        fl2 = FrameLogger("&&& After LUCA")
        fl3 = FrameLogger("$$$ after fcp, before tts")
        luca = LLMUserContextAggregator(context)
        laca = LLMAssistantContextAggregator(context)
        ta = TalkingAnimation()

        pipeline = Pipeline([
            transport.input(),
            luca,
            fl2,
            llm,
            fcp,
            fl3,
            tts,
            ta,
            transport.output(),
            laca
        ])

        task = PipelineTask(pipeline, allow_interruptions=True)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            print(f"Context is: {context}")
            await task.queue_frames([OpenAILLMContextFrame(context)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
