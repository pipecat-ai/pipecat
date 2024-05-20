import asyncio
import aiohttp
import copy
import json
import os
import re
import sys
import wave
from typing import List

from openai._types import NotGiven, NOT_GIVEN

from openai.types.chat import (
    ChatCompletionToolParam,
)

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator, LLMAssistantResponseAggregator
from pipecat.processors.logger import FrameLogger
from pipecat.frames.frames import (
    Frame,
    LLMMessagesFrame,
    LLMFunctionCallFrame,
    LLMFunctionStartFrame,
    AudioRawFrame,
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
        sounds[file] = audio_file.readframes(-1)

steps = [{"prompt": "Start by introducing yourself. Then, ask the user to confirm their identity by telling you their birthday, including the year. When they answer with their birthday, call the verify_birthday function.",
          "run_async": False,
          "failed": "The user provided an incorrect birthday. Ask them for their birthday again. When they answer, call the verify_birthday function.",
          "tools": [{"type": "function",
                     "function": {"name": "verify_birthday",
                                  "description": "Use this function to verify the user has provided their correct birthday.",
                                  "parameters": {"type": "object",
                                                 "properties": {"birthday": {"type": "string",
                                                                             "description": "The user's birthdate, including the year. The user can provide it in any format, but convert it to YYYY-MM-DD format to call this function.",
                                                                             }},
                                                 },
                                  },
                     }],
          },
         {"prompt": "Next, thank the user for confirming their identity, then ask the user to list their current prescriptions. Each prescription needs to have a medication name and a dosage. Do not call the list_prescriptions function with any unknown dosages.",
          "run_async": True,
          "tools": [{"type": "function",
                     "function": {"name": "list_prescriptions",
                                  "description": "Once the user has provided a list of their prescription medications, call this function.",
                                  "parameters": {"type": "object",
                                                 "properties": {"prescriptions": {"type": "array",
                                                                                  "items": {"type": "object",
                                                                                            "properties": {"medication": {"type": "string",
                                                                                                                          "description": "The medication's name",
                                                                                                                          },
                                                                                                           "dosage": {"type": "string",
                                                                                                                      "description": "The prescription's dosage",
                                                                                                                      },
                                                                                                           },
                                                                                            },
                                                                                  }},
                                                 },
                                  },
                     }],
          },
         {"prompt": "Next, ask the user if they have any allergies. Once they have listed their allergies or confirmed they don't have any, call the list_allergies function.",
          "run_async": True,
          "tools": [{"type": "function",
                     "function": {"name": "list_allergies",
                                  "description": "Once the user has provided a list of their allergies, call this function.",
                                  "parameters": {"type": "object",
                                                 "properties": {"allergies": {"type": "array",
                                                                              "items": {"type": "object",
                                                                                        "properties": {"name": {"type": "string",
                                                                                                                "description": "What the user is allergic to",
                                                                                                                }},
                                                                                        },
                                                                              }},
                                                 },
                                  },
                     }],
          },
         {"prompt": "Now ask the user if they have any medical conditions the doctor should know about. Once they've answered the question, call the list_conditions function.",
          "run_async": True,
          "tools": [{"type": "function",
                     "function": {"name": "list_conditions",
                                  "description": "Once the user has provided a list of their medical conditions, call this function.",
                                  "parameters": {"type": "object",
                                                 "properties": {"conditions": {"type": "array",
                                                                               "items": {"type": "object",
                                                                                         "properties": {"name": {"type": "string",
                                                                                                                 "description": "The user's medical condition",
                                                                                                                 }},
                                                                                         },
                                                                               }},
                                                 },
                                  },
                     },
                    ],
          },
         {"prompt": "Finally, ask the user the reason for their doctor visit today. Once they answer, call the list_visit_reasons function.",
          "run_async": True,
          "tools": [{"type": "function",
                     "function": {"name": "list_visit_reasons",
                                  "description": "Once the user has provided a list of the reasons they are visiting a doctor today, call this function.",
                                  "parameters": {"type": "object",
                                                 "properties": {"visit_reasons": {"type": "array",
                                                                                  "items": {"type": "object",
                                                                                            "properties": {"name": {"type": "string",
                                                                                                                    "description": "The user's reason for visiting the doctor",
                                                                                                                    }},
                                                                                            },
                                                                                  }},
                                                 },
                                  },
                     }],
          },
         {"prompt": "Now, thank the user and end the conversation.",
          "run_async": True,
          "tools": [],
          },
         {"prompt": "",
          "run_async": True,
          "tools": []},
         ]
current_step = 0


class ChecklistProcessor(FrameProcessor):

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
        self._id = "You are Jessica, an agent for a company called Tri-County Health Services. Your job is to collect important information from the user before their doctor visit. You're talking to Chad Bailey. You should address the user by their first name and be polite and professional. You're not a medical professional, so you shouldn't provide any advice. Keep your responses short. Your job is to collect information to give to a doctor. Don't make assumptions about what values to plug into functions. Ask for clarification if a user response is ambiguous."

        # Create an allowlist of functions that the LLM can call
        self._functions = [
            "verify_birthday",
            "list_prescriptions",
            "list_allergies",
            "list_conditions",
            "list_visit_reasons",
        ]

        self._context.add_message(
            {"role": "system", "content": f"{self._id} {steps[0]['prompt']}"}
        )

        if tools:
            self._context.set_tools(tools)

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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        global current_step
        this_step = steps[current_step]
        self._context.set_tools(this_step["tools"])
        print(f"tools are: {self._context.tools}")
        if isinstance(frame, LLMFunctionStartFrame):
            print(f"... Preparing function call: {frame.function_name}")
            self._function_name = frame.function_name
            if this_step["run_async"]:
                # Get the LLM talking about the next step before getting the rest
                # of the function call completion
                current_step += 1
                self._context.add_message(
                    {"role": "system", "content": steps[current_step]["prompt"]}
                )
                await self.push_frame(OpenAILLMContextFrame(self._context))

                local_context = copy.deepcopy(self._context)
                local_context.set_tool_choice("none")
                async for frame in self._llm.process_frame(
                    OpenAILLMContextFrame(local_context)
                ):
                    await self.push_frame(frame)
            else:
                # Insert a quick response while we run the function
                await self.push_frame(AudioRawFrame(sounds["ding2.wav"]))
                pass
        elif isinstance(frame, LLMFunctionCallFrame):

            if frame.function_name and frame.arguments:
                print(
                    f"--> Calling function: {frame.function_name} with arguments:")
                pretty_json = re.sub(
                    "\n", "\n    ", json.dumps(
                        json.loads(
                            frame.arguments), indent=2))
                print(f"--> {pretty_json}\n")
                if frame.function_name not in self._functions:
                    raise Exception(
                        f"The LLM tried to call a function named {frame.function_name}, which isn't in the list of known functions. Please check your prompt and/or self._functions."
                    )
                fn = getattr(self, frame.function_name)
                result = fn(json.loads(frame.arguments))

                if not this_step["run_async"]:
                    if result:
                        current_step += 1
                        self._context.add_message(
                            {"role": "system", "content": steps[current_step]["prompt"]}
                        )
                        await self.push_frame(OpenAILLMContextFrame(self._context))

                        local_context = copy.deepcopy(self._context)
                        local_context.set_tool_choice("none")
                        async for frame in self._llm.process_frame(
                            OpenAILLMContextFrame(local_context)
                        ):
                            await self.push_frame(frame)
                    else:
                        self._context.add_message(
                            {"role": "system", "content": this_step["failed"]}
                        )
                        await self.push_frame(OpenAILLMContextFrame(self._context))

                        local_context = copy.deepcopy(self._context)
                        local_context.set_tool_choice("none")
                        async for frame in self._llm.process_frame(
                            OpenAILLMContextFrame(local_context)
                        ):
                            await self.push_frame(frame)
                    print(f"<-- Verify result: {result}\n")

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
            model="gpt-4-turbo")

        messages = []
        context = OpenAILLMContext(
            messages=messages,
        )
        user_response = LLMUserResponseAggregator(messages)
        assistant_response = LLMAssistantResponseAggregator(messages)
        checklist = ChecklistProcessor(context, llm)
        fl = FrameLogger("after transport output")
        pipeline = Pipeline([
            transport.input(),
            user_response,
            llm,
            checklist,
            tts,
            transport.output(),
            assistant_response,
            fl
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
