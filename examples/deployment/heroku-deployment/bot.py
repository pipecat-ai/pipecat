import asyncio
import os
import sys
import wave

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.logger import FrameLogger
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMContext, OpenAILLMContextFrame, OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper

# Load environment variables
load_dotenv(override=True)

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Load sound files
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
    full_path = os.path.join(script_dir, "assets", file)
    filename = os.path.splitext(os.path.basename(full_path))[0]
    with wave.open(full_path) as audio_file:
        sounds[file] = OutputAudioRawFrame(
            audio_file.readframes(-1), audio_file.getframerate(), audio_file.getnchannels()
        )


class IntakeProcessor:
    def __init__(self, context: OpenAILLMContext):
        print("Initializing context from IntakeProcessor")
        context.add_message(
            {
                "role": "system",
                "content": "You are Jessica, an agent for a company called Tri-County Health Services. Your job is to collect important information from the user before their doctor visit. You're talking to Chad Bailey. You should address the user by their first name and be polite and professional. You're not a medical professional, so you shouldn't provide any advice. Keep your responses short. Your job is to collect information to give to a doctor. Don't make assumptions about what values to plug into functions. Ask for clarification if a user response is ambiguous. Start by introducing yourself. Then, ask the user to confirm their identity by telling you their birthday, including the year. When they answer with their birthday, call the verify_birthday function.",
            }
        )
        context.set_tools(
            [
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
            ]
        )

    async def verify_birthday(
        self, function_name, tool_call_id, args, llm, context, result_callback
    ):
        if args["birthday"] == "1983-01-01":
            context.set_tools(
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
                                    }
                                },
                            },
                        },
                    }
                ]
            )
            # It's a bit weird to push this to the LLM, but it gets it into the pipeline
            # await llm.push_frame(sounds["ding2.wav"], FrameDirection.DOWNSTREAM)
            # We don't need the function call in the context, so just return a new
            # system message and let the framework re-prompt
            await result_callback(
                [
                    {
                        "role": "system",
                        "content": "Next, thank the user for confirming their identity, then ask the user to list their current prescriptions. Each prescription needs to have a medication name and a dosage. Do not call the list_prescriptions function with any unknown dosages.",
                    }
                ]
            )
        else:
            # The user provided an incorrect birthday; ask them to try again
            await result_callback(
                [
                    {
                        "role": "system",
                        "content": "The user provided an incorrect birthday. Ask them for their birthday again. When they answer, call the verify_birthday function.",
                    }
                ]
            )

    async def start_prescriptions(self, function_name, llm, context):
        print("!!! doing start prescriptions")
        # Move on to allergies
        context.set_tools(
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
                                            }
                                        },
                                    },
                                }
                            },
                        },
                    },
                }
            ]
        )
        context.add_message(
            {
                "role": "system",
                "content": "Next, ask the user if they have any allergies. Once they have listed their allergies or confirmed they don't have any, call the list_allergies function.",
            }
        )
        print("!!! about to await llm process frame in start prescrpitions")
        await llm.queue_frame(OpenAILLMContextFrame(context), FrameDirection.DOWNSTREAM)
        print("!!! past await process frame in start prescriptions")

    async def start_allergies(self, function_name, llm, context):
        print("!!! doing start allergies")
        # Move on to conditions
        context.set_tools(
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
                                            }
                                        },
                                    },
                                }
                            },
                        },
                    },
                },
            ]
        )
        context.add_message(
            {
                "role": "system",
                "content": "Now ask the user if they have any medical conditions the doctor should know about. Once they've answered the question, call the list_conditions function.",
            }
        )
        await llm.queue_frame(OpenAILLMContextFrame(context), FrameDirection.DOWNSTREAM)

    async def start_conditions(self, function_name, llm, context):
        print("!!! doing start conditions")
        # Move on to visit reasons
        context.set_tools(
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
                                            }
                                        },
                                    },
                                }
                            },
                        },
                    },
                }
            ]
        )
        context.add_message(
            {
                "role": "system",
                "content": "Finally, ask the user the reason for their doctor visit today. Once they answer, call the list_visit_reasons function.",
            }
        )
        await llm.queue_frame(OpenAILLMContextFrame(context), FrameDirection.DOWNSTREAM)

    async def start_visit_reasons(self, function_name, llm, context):
        print("!!! doing start visit reasons")
        # move to finish call
        context.set_tools([])
        context.add_message(
            {"role": "system", "content": "Now, thank the user and end the conversation."}
        )
        await llm.queue_frame(OpenAILLMContextFrame(context), FrameDirection.DOWNSTREAM)

    async def save_data(self, function_name, tool_call_id, args, llm, context, result_callback):
        logger.info(f"!!! Saving data: {args}")
        # Since this is supposed to be "async", returning None from the callback
        # will prevent adding anything to context or re-prompting
        await result_callback(None)


async def get_room_config(session: aiohttp.ClientSession):
    """Get Daily room configuration from environment variables"""
    url = os.getenv("DAILY_ROOM_URL")
    api_key = os.getenv("DAILY_API_KEY")

    if not url:
        raise Exception("DAILY_ROOM_URL environment variable is required")
    if not api_key:
        raise Exception("DAILY_API_KEY environment variable is required")

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=api_key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=session,
    )

    # Create token with 1 hour expiry
    token = await daily_rest_helper.get_token(url, 60 * 60)

    return url, token


async def main():
    # Check for required environment variables
    required_vars = ["DAILY_ROOM_URL", "DAILY_API_KEY", "CARTESIA_API_KEY", "OPENAI_API_KEY"]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")

    async with aiohttp.ClientSession() as session:
        room_url, token = await get_room_config(session)

        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

        messages = []
        context = OpenAILLMContext(messages=messages)
        context_aggregator = llm.create_context_aggregator(context)

        intake = IntakeProcessor(context)
        llm.register_function("verify_birthday", intake.verify_birthday)
        llm.register_function(
            "list_prescriptions", intake.save_data, start_callback=intake.start_prescriptions
        )
        llm.register_function(
            "list_allergies", intake.save_data, start_callback=intake.start_allergies
        )
        llm.register_function(
            "list_conditions", intake.save_data, start_callback=intake.start_conditions
        )
        llm.register_function(
            "list_visit_reasons", intake.save_data, start_callback=intake.start_visit_reasons
        )

        fl = FrameLogger("LLM Output")

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm,
                fl,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=False))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            print(f"Context is: {context}")
            await task.queue_frames([OpenAILLMContextFrame(context)])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())