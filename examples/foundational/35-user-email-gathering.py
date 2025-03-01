#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMContext, OpenAILLMService
from pipecat.services.rime import RimeHttpTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def store_user_emails(function_name, tool_call_id, args, llm, context, result_callback):
    print(f"User emails: {args}")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        # Cartesia offers a `<spell></spell>` tags that we can use to ask the user
        # to confirm the emails.
        # (see https://docs.cartesia.ai/build-with-sonic/formatting-text-for-sonic/spelling-out-input-text)
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
            aiohttp_session=session,
            eos_skip_tags=[("<spell>", "</spell>")],
        )

        # Rime offers a function `spell()` that we can use to ask the user
        # to confirm the emails.
        # (see https://docs.rime.ai/api-reference/spell)
        # tts = RimeHttpTTSService(
        #     api_key=os.getenv("RIME_API_KEY", ""),
        #     voice_id="eva",
        #     aiohttp_session=session,
        #     eos_skip_tags=[("spell(", ")")],
        # )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
        # Register a function_name of None to get all functions
        # sent to the same callback with an additional function_name parameter.
        llm.register_function(None, store_user_emails)

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

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
