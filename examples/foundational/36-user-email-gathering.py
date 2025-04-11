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
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.rime.tts import RimeHttpTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)


async def store_user_emails(function_name, tool_call_id, args, llm, context, result_callback):
    print(f"User emails: {args}")


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    logger.info(f"Starting bot")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

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

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
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
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Start conversation - empty prompt to let LLM follow system instructions
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")

    @transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info(f"Client closed connection")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    from run import main

    main()
