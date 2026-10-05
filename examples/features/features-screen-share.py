#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A bot that can look at the user's camera or screen share when asked.

The transport captures both video sources as the client connects, listed in
``video_in_sources``. Each is captured with ``on_request_only=True``, so no
frames flow until the bot asks for one: the LLM has a tool per source,
``fetch_camera_image`` and ``fetch_screen_share_image``, and each pushes a
``UserImageRequestFrame`` for its source. The transport answers with the next
frame from that source, which is added to the LLM context so the LLM can
describe it.

Share your screen from the client and ask what's on it, or turn on your camera
and ask what it sees.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame, UserImageRequestFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import (
    create_transport,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams, VideoInSourceParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


async def request_user_image(
    params: FunctionCallParams, user_id: str, question: str, video_source: str
):
    """Request an image from the user's video and add it to the LLM context.

    This pushes a UserImageRequestFrame upstream to the transport. The transport
    answers it with the next frame from the requested video source, as a
    UserImageRawFrame that the LLM assistant aggregator adds to the context. The
    result_callback is invoked once the image is retrieved and processed.
    """
    logger.debug(f"Requesting {video_source} image with user_id={user_id}, question={question}")

    await params.llm.push_frame(
        UserImageRequestFrame(
            user_id=user_id,
            text=question,
            video_source=video_source,
            append_to_context=True,
            function_name=params.function_name,
            tool_call_id=params.tool_call_id,
            result_callback=params.result_callback,
        ),
        FrameDirection.UPSTREAM,
    )


async def fetch_camera_image(params: FunctionCallParams, user_id: str, question: str):
    """Get an image from the user's camera to answer a question about it.

    Args:
        user_id: The ID of the user to grab the image from.
        question: The question that the user is asking about the image.
    """
    await request_user_image(params, user_id, question, "camera")


async def fetch_screen_share_image(params: FunctionCallParams, user_id: str, question: str):
    """Get an image of the user's screen share to answer a question about it.

    Args:
        user_id: The ID of the user to grab the image from.
        question: The question that the user is asking about the image.
    """
    await request_user_image(params, user_id, question, "screenVideo")


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
        video_in_sources={
            "camera": VideoInSourceParams(on_request_only=True),
            "screenVideo": VideoInSourceParams(on_request_only=True),
        },
    ),
    "livekit": lambda: LiveKitParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        video_in_sources={
            "camera": VideoInSourceParams(on_request_only=True),
            "screenVideo": VideoInSourceParams(on_request_only=True),
        },
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        video_in_sources={
            "camera": VideoInSourceParams(on_request_only=True),
            "screenVideo": VideoInSourceParams(on_request_only=True),
        },
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="86e30c1d-714b-4074-a1f2-1cb6b552fb49",
        ),
    )

    llm = OpenAIResponsesLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIResponsesLLMService.Settings(
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way. You are able to describe images from the user's camera and screen share.",
        ),
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check on that.", append_to_context=False))

    context = LLMContext(tools=[fetch_camera_image, fetch_screen_share_image])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

        client_id = transport.get_client_id(client)

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
        logger.info("Client disconnected")
        await runner.cancel()

    @tts.event_handler("on_tts_request")
    async def on_tts_request(tts, context_id: str, text: str):
        logger.debug(f"On TTS request: {context_id}: {text}")

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
