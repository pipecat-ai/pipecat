#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService, LLMSearchResponseFrame
from pipecat.services.llm_service import LLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv(override=True)


# Function handlers for the LLM
search_tool = {"google_search": {}}
tools = [search_tool]

system_instruction = """
You are an expert at providing the most recent news from any place. Your responses will be converted to audio, so avoid using special characters or overly complex formatting.

Always use the google search API to retrieve the latest news. You must also use it to check which day is today.

You can:
- Use the Google search API to check the current date.
- Provide the most recent and relevant news from any place by using the google search API.
- Answer any questions the user may have, ensuring your responses are accurate and concise.

Start each interaction by asking the user about which place they would like to know the information.
"""


class LLMSearchLoggerObserver(BaseObserver):
    async def on_push_frame(self, data: FramePushed):
        src = data.source
        dst = data.destination
        frame = data.frame
        timestamp = data.timestamp

        if not isinstance(src, LLMService) and not isinstance(dst, LLMService):
            return

        time_sec = timestamp / 1_000_000_000

        arrow = "→"

        if isinstance(frame, LLMSearchResponseFrame):
            logger.debug(f"🧠 {arrow} {dst} LLM SEARCH RESPONSE FRAME: {frame} at {time_sec:.2f}s")


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # Initialize the Gemini Multimodal Live model
    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        tools=tools,
    )

    context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": "Start by greeting the user warmly, introducing yourself, and mentioning the current day. Be friendly and engaging to set a positive tone for the interaction.",
            }
        ],
    )
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[LLMSearchLoggerObserver()],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Start conversation - empty prompt to let LLM follow system instructions
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
