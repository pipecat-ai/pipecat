#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.processors.frameworks.langchain import LangchainProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from loguru import logger

from runner import configure

from dotenv import load_dotenv

load_dotenv(override=True)


logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

message_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]


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

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Be nice and helpful. Answer very briefly and without special characters like `#` or `*`. "
                    "Your response will be synthesized to voice and those characters will create unnatural sounds.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        chain = prompt | ChatOpenAI(model="gpt-4o", temperature=0.7)
        history_chain = RunnableWithMessageHistory(
            chain,
            get_session_history,
            history_messages_key="chat_history",
            input_messages_key="input",
        )
        lc = LangchainProcessor(history_chain)

        tma_in = LLMUserResponseAggregator()
        tma_out = LLMAssistantResponseAggregator()

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                tma_in,  # User responses
                lc,  # Langchain
                tts,  # TTS
                transport.output(),  # Transport bot output
                tma_out,  # Assistant spoken responses
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            lc.set_participant_id(participant["id"])
            # Kick off the conversation.
            # the `LLMMessagesFrame` will be picked up by the LangchainProcessor using
            # only the content of the last message to inject it in the prompt defined
            # above. So no role is required here.
            messages = [({"content": "Please briefly introduce yourself to the user."})]
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
