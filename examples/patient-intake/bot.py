#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys
import wave

from typing import List

from openai._types import NotGiven, NOT_GIVEN

from openai.types.chat import (
    ChatCompletionToolParam,
)

from pipecat.frames.frames import AudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator, LLMAssistantContextAggregator
from pipecat.processors.logger import FrameLogger
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMContext, OpenAILLMContextFrame, OpenAILLMService
from pipecat.services.ai_services import AIService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sounds = {}
sound_files = [
    "clack-short.wav",
    "clack.wav",
    "clack-short-quiet.wav",
    "ding.wav",
    "ding2.wav",
]

script_dir = os.path.dirname(__file__)

for file in sound_files:
    # Build the full path to the sound file
    full_path = os.path.join(script_dir, "assets", file)
    # Get the filename without the extension to use as the dictionary key
    filename = os.path.splitext(os.path.basename(full_path))[0]
    # Open the sound and convert it to bytes
    with wave.open(full_path) as audio_file:
        sounds[file] = AudioRawFrame(audio_file.readframes(-1),
                                     audio_file.getframerate(), audio_file.getnchannels())


class IntakeProcessor:
    def __init__(
        self,
        context: OpenAILLMContext,
        llm: AIService,
        tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._context: OpenAILLMContext = context
        self._llm = llm
        print(f"Initializing context from IntakeProcessor")
        self._context.add_message({"role": "system", "content": "You are Jessica, an agent for a company called Tri-County Health Services. Your job is to collect important information from the user before their doctor visit. You're talking to Chad Bailey. You should address the user by their first name and be polite and professional. You're not a medical professional, so you shouldn't provide any advice. Keep your responses short. Your job is to collect information to give to a doctor. Don't make assumptions about what values to plug into functions. Ask for clarification if a user response is ambiguous. Start by introducing yourself. Then, ask the user to confirm their identity by telling you their birthday, including the year. When they answer with their birthday, call the verify_birthday function."})
        self._context.set_tools([
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
                            }},
                    },
                },
            }])
        # Create an allowlist of functions that the LLM can call
        self._functions = [
            "verify_birthday",
            "list_prescriptions",
            "list_allergies",
            "list_conditions",
            "list_visit_reasons",
        ]

    async def verify_birthday(self, llm, args):
        if args["birthday"] == "1983-01-01":
            self._context.set_tools(
                [
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
                                    }},
                            },
                        },
                    }])
            # It's a bit weird to push this to the LLM, but it gets it into the pipeline
            await llm.push_frame(sounds["ding2.wav"], FrameDirection.DOWNSTREAM)
            # We don't need the function call in the context, so just return a new
            # system message and let the framework re-prompt
            return [{"role": "system", "content": "Next, thank the user for confirming their identity, then ask the user to list their current prescriptions. Each prescription needs to have a medication name and a dosage. Do not call the list_prescriptions function with any unknown dosages."}]
        else:
            # The user provided an incorrect birthday; ask them to try again
            return [{"role": "system", "content": "The user provided an incorrect birthday. Ask them for their birthday again. When they answer, call the verify_birthday function."}]

    async def start_prescriptions(self, llm):
        print(f"!!! doing start prescriptions")
        # Move on to allergies
        self._context.set_tools(
            [
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
                                            }},
                                    },
                                }},
                        },
                    },
                }])
        self._context.add_message(
            {
                "role": "system",
                "content": "Next, ask the user if they have any allergies. Once they have listed their allergies or confirmed they don't have any, call the list_allergies function."})
        print(f"!!! about to await llm process frame in start prescrpitions")
        await llm.process_frame(OpenAILLMContextFrame(self._context), FrameDirection.DOWNSTREAM)
        print(f"!!! past await process frame in start prescriptions")

    async def start_allergies(self, llm):
        print("!!! doing start allergies")
        # Move on to conditions
        self._context.set_tools(
            [
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
                                            }},
                                    },
                                }},
                        },
                    },
                },
            ])
        self._context.add_message(
            {
                "role": "system",
                "content": "Now ask the user if they have any medical conditions the doctor should know about. Once they've answered the question, call the list_conditions function."})
        await llm.process_frame(OpenAILLMContextFrame(self._context), FrameDirection.DOWNSTREAM)

    async def start_conditions(self, llm):
        print("!!! doing start conditions")
        # Move on to visit reasons
        self._context.set_tools(
            [
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
                                            }},
                                    },
                                }},
                        },
                    },
                }])
        self._context.add_message(
            {"role": "system", "content": "Finally, ask the user the reason for their doctor visit today. Once they answer, call the list_visit_reasons function."})
        await llm.process_frame(OpenAILLMContextFrame(self._context), FrameDirection.DOWNSTREAM)

    async def start_visit_reasons(self, llm):
        print("!!! doing start visit reasons")
        # move to finish call
        self._context.set_tools([])
        self._context.add_message({"role": "system",
                                   "content": "Now, thank the user and end the conversation."})
        await llm.process_frame(OpenAILLMContextFrame(self._context), FrameDirection.DOWNSTREAM)

    async def save_data(self, llm, args):
        logger.info(f"!!! Saving data: {args}")
        # Since this is supposed to be "async", returning None from the callback
        # will prevent adding anything to context or re-prompting
        return None


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
                #
                # Spanish
                #
                # transcription_settings=DailyTranscriptionSettings(
                #     language="es",
                #     tier="nova",
                #     model="2-general"
                # )
            )
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            #
            # English
            #
            voice_id="pNInz6obpgDQGcFmaJgB",

            #
            # Spanish
            #
            # model="eleven_multilingual_v2",
            # voice_id="gD1IexrzCvsXPHUuT0s3",
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o")

        messages = []
        context = OpenAILLMContext(messages=messages)
        user_context = LLMUserContextAggregator(context)
        assistant_context = LLMAssistantContextAggregator(context)

        intake = IntakeProcessor(context, llm)
        llm.register_function("verify_birthday", intake.verify_birthday)
        llm.register_function(
            "list_prescriptions",
            intake.save_data,
            start_callback=intake.start_prescriptions)
        llm.register_function(
            "list_allergies",
            intake.save_data,
            start_callback=intake.start_allergies)
        llm.register_function(
            "list_conditions",
            intake.save_data,
            start_callback=intake.start_conditions)
        llm.register_function(
            "list_visit_reasons",
            intake.save_data,
            start_callback=intake.start_visit_reasons)

        fl = FrameLogger("LLM Output")

        pipeline = Pipeline([
            transport.input(),   # Transport input
            user_context,        # User responses
            llm,                 # LLM
            fl,                  # Frame logger
            tts,                 # TTS
            transport.output(),  # Transport output
            assistant_context,   # Assistant responses
        ])

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=False))

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
