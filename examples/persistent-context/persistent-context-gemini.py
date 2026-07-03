#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import glob
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame, UserImageRequestFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import (
    create_transport,
    get_transport_client_id,
    maybe_capture_participant_camera,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


BASE_FILENAME = "/tmp/pipecat_conversation_"


async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    temperature = 75 if format == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": format,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def get_image(params: FunctionCallParams, user_id: str, question: str):
    """Called when the user requests a description of their camera feed.

    Args:
        user_id: The ID of the user to grab the image from.
        question: The question that the user is asking about the image.
    """
    logger.debug(f"Requesting image with user_id={user_id}, question={question}")

    # Request a user image frame and indicate that it should be added to the
    # context. Also associate it to the function call. Pass the result_callback
    # so it can be invoked when the image is actually retrieved.
    await params.llm.push_frame(
        UserImageRequestFrame(
            user_id=user_id,
            text=question,
            append_to_context=True,
            function_name=params.function_name,
            tool_call_id=params.tool_call_id,
            result_callback=params.result_callback,
        ),
        FrameDirection.UPSTREAM,
    )


async def get_saved_conversation_filenames(params: FunctionCallParams):
    """Get a list of saved conversation histories. Returns a list of filenames. Each filename includes a date and timestamp. Each file is conversation history that can be loaded into this session."""
    # Construct the full pattern including the BASE_FILENAME
    full_pattern = f"{BASE_FILENAME}*.json"

    # Use glob to find all matching files
    matching_files = glob.glob(full_pattern)
    logger.debug(f"matching files: {matching_files}")

    await params.result_callback({"filenames": matching_files})


async def save_conversation(params: FunctionCallParams):
    """Save the current conversation. Use this function to persist the current conversation to external storage."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filename = f"{BASE_FILENAME}{timestamp}.json"
    logger.debug(
        f"writing conversation to {filename}\n{json.dumps(params.context.get_messages(), indent=4)}"
    )
    try:
        with open(filename, "w") as file:
            messages = params.context.get_messages()
            # remove the last message (the instruction to save the context)
            messages.pop()
            json.dump(messages, file, indent=2)
        await params.result_callback({"success": True})
    except Exception as e:
        logger.debug(f"error saving conversation: {e}")
        await params.result_callback({"success": False, "error": str(e)})


async def load_conversation(params: FunctionCallParams, filename: str):
    """Load a conversation history. Use this function to load a conversation history into the current session.

    Args:
        filename: The filename of the conversation history to load.
    """
    logger.debug(f"loading conversation from {filename}")
    try:
        with open(filename) as file:
            params.context.set_messages(json.load(file))
        await params.result_callback(
            {
                "success": True,
                "message": "The most recent conversation has been loaded. Awaiting further instructions.",
            }
        )
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


system_instruction = """You are a helpful assistant in a voice conversation. Your goal is to demonstrate your
capabilities in a succinct way. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Keep responses concise. Respond to what the user said in a creative
can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative
and helpful way.

You have several tools you can use to help you.

You can respond to questions about the weather using the get_current_weather tool.

You can save the current conversation using the save_conversation tool. This tool allows you to save
the current conversation to external storage. If the user asks you to save the conversation, use this
save_conversation too.

You can load a saved conversation using the load_conversation tool. This tool allows you to load a
conversation from external storage. You can get a list of conversations that have been saved using the
get_saved_conversation_filenames tool.

You can answer questions about the user's video stream using the get_image tool. Some examples of phrases that \
indicate you should use the get_image tool are:
  - What do you see?
  - What's in the video?
  - Can you describe the video?
  - Tell me about what you see.
  - Tell me something interesting about what you see.
  - What's happening in the video?
"""


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = GoogleLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        system_instruction=system_instruction,
    )

    context = LLMContext(
        tools=[
            get_current_weather,
            save_conversation,
            get_saved_conversation_filenames,
            load_conversation,
            get_image,
        ]
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            user_aggregator,
            llm,  # LLM
            tts,
            transport.output(),  # Transport bot output
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")

        await maybe_capture_participant_camera(transport, client)

        client_id = get_transport_client_id(transport, client)

        # Kick off the conversation.
        context.add_message(
            {
                "role": "developer",
                "content": f"Please introduce yourself to the user. Use '{client_id}' as the user ID during function calls.",
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
