#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os

from dotenv import load_dotenv
from loguru import logger
from openai.types.chat import ChatCompletionToolParam

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


class SwitchVoices(ParallelPipeline):
    def __init__(self):
        self._current_voice = "News Lady"

        news_lady = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="bf991597-6c13-47e4-8411-91ec2de5c466",  # Newslady
        )

        british_lady = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        barbershop_man = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # Barbershop Man
        )

        super().__init__(
            # News Lady voice
            [FunctionFilter(self.news_lady_filter), news_lady],
            # British Reading Lady voice
            [FunctionFilter(self.british_lady_filter), british_lady],
            # Barbershop Man voice
            [FunctionFilter(self.barbershop_man_filter), barbershop_man],
        )

    @property
    def current_voice(self):
        return self._current_voice

    async def switch_voice(self, params: FunctionCallParams):
        self._current_voice = params.arguments["voice"]
        await params.result_callback(
            {
                "voice": f"You are now using your {self.current_voice} voice. Your responses should now be as if you were a {self.current_voice}."
            }
        )

    async def news_lady_filter(self, _: Frame) -> bool:
        return self.current_voice == "News Lady"

    async def british_lady_filter(self, _: Frame) -> bool:
        return self.current_voice == "British Lady"

    async def barbershop_man_filter(self, _: Frame) -> bool:
        return self.current_voice == "Barbershop Man"


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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = SwitchVoices()

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    llm.register_function("switch_voice", tts.switch_voice)

    tools = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "switch_voice",
                "description": "Switch your voice only when the user asks you to",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "voice": {
                            "type": "string",
                            "description": "The voice the user wants you to use",
                        },
                    },
                    "required": ["voice"],
                },
            },
        )
    ]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities. Respond to what the user said in a creative and helpful way. Your output should not include non-alphanumeric characters. You can do the following voices: 'News Lady', 'British Lady' and 'Barbershop Man'.",
        },
    ]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS with switch voice functionality
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
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
        # Kick off the conversation.
        messages.append(
            {
                "role": "system",
                "content": f"Please introduce yourself to the user and let them know the voices you can do. Your initial responses should be as if you were a {tts.current_voice}.",
            }
        )
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
