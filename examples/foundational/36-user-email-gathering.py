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
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.rime.tts import RimeHttpTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


async def store_user_emails(params: FunctionCallParams):
    print(f"User emails: {params.arguments}")


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

    # Cartesia offers a `<spell></spell>` tags that we can use to ask the user
    # to confirm the emails.
    # (see https://docs.cartesia.ai/build-with-sonic/formatting-text-for-sonic/spelling-out-input-text)
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # Rime offers a function `spell()` that we can use to ask the user
    # to confirm the emails.
    # (see https://docs.rime.ai/api-reference/spell)
    # tts = RimeHttpTTSService(
    #     api_key=os.getenv("RIME_API_KEY", ""),
    #     voice_id="eva",
    #     aiohttp_session=session,
    # )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    # You can aslo register a function_name of None to get all functions
    # sent to the same callback with an additional function_name parameter.
    llm.register_function("store_user_emails", store_user_emails)

    tools = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "store_user_emails",
                "description": "Store user emails when confirmed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "emails": {
                            "type": "array",
                            "description": "The list of user emails",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["emails"],
                },
            },
        )
    ]
    messages = [
        {
            "role": "system",
            # Cartesia <spell></spell>
            "content": "You need to gather a valid email or emails from the user. Your output will be converted to audio so don't include special characters in your answers. If the user provides one or more email addresses confirm them with the user. Enclose all emails with <spell> tags, for example <spell>a@a.com</spell>.",
            # Rime spell()
            # "content": "You need to gather a valid email or emails from the user. Your output will be converted to audio so don't include special characters in your answers. If the user provides one or more email addresses confirm them with the user. Enclose all emails with spell(), for example spell(a@a.com).",
        },
    ]

    context = OpenAILLMContext(messages, tools)
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
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Start conversation - empty prompt to let LLM follow system instructions
        await task.queue_frames([LLMRunFrame()])

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
