#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Podcast Interview Example.

This example demonstrates a podcast interview flow using Pipecat Flows.

The flow handles:
1. Introduction and guest introduction
2. Topic selection
3. Interview with multiple questions and follow-ups
4. Conclusion and wrap-up
5. Final thank you

Multi-LLM Support:
Set LLM_PROVIDER environment variable to choose your LLM provider.
Supported: openai_responses (default), openai, anthropic, google, aws

Requirements:
- CARTESIA_API_KEY (for TTS)
- DEEPGRAM_API_KEY (for STT)
- DAILY_API_KEY (for transport)
- LLM API key (varies by provider - see env.example)
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from loguru import logger
from utils import create_llm

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.flows import FlowManager, NodeConfig
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    # Behavioral evals: run with `-t eval` to drive this bot via `pipecat eval`.
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# Type definitions
class ProceedToTopicResult(TypedDict):
    """Result type for proceed_to_topic function"""

    guest_summary: str


class StartInterviewResult(TypedDict):
    """Result type for start_interview function"""

    topic: str


# Functions for Introduction Node
async def proceed_to_topic(
    flow_manager: FlowManager, guest_summary: str
) -> tuple[ProceedToTopicResult | None, NodeConfig]:
    """Use after the guest has introduced themselves.

    Args:
        guest_summary (str): A quick summary of who the guest is (name, role, area of expertise, etc.).
    """
    return ProceedToTopicResult(guest_summary=guest_summary), create_topic_node()


# Functions for Topic Node
async def start_interview(
    flow_manager: FlowManager, topic: str
) -> tuple[StartInterviewResult | None, NodeConfig]:
    """Use this when the guest has shared a clear topic they want to explore.

    Args:
        topic (str): The topic the guest wants to discuss.
    """
    return StartInterviewResult(topic=topic), create_interview_node()


# Functions for Interview Node
async def next_question(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """Use this after you've thoroughly explored the current aspect with multiple questions and follow-ups."""
    return None, create_interview_node()


async def wrap_up(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """Use this when you've gathered substantial insights and are ready to wrap up."""
    return None, create_conclusion_node()


# Functions for Conclusion Node
async def end_interview(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """Use this after the guest has shared their final thoughts."""
    return None, create_final_node()


def create_introduction_node() -> NodeConfig:
    """Create the Introduction node."""
    return NodeConfig(
        name="introduction",
        role_message="You are a warm, engaging podcast host with a natural conversational style. You're genuinely curious about your guests and skilled at making them feel comfortable while drawing out interesting insights. Your questions flow naturally, and you listen actively, building on what your guest shares.",
        task_messages=[
            {
                "role": "developer",
                "content": "Welcome the guest warmly and enthusiastically. Focus this exchange on getting to know who they are. Invite them to briefly introduce themselves—name, role, current focus, or anything fun they'd like to share. Ask one follow-up question if it helps clarify or highlight something interesting about them. Once you feel you have a clear introduction, use the proceed_to_topic function to move into topic selection.",
            }
        ],
        functions=[proceed_to_topic],
    )


def create_topic_node() -> NodeConfig:
    """Create the Topic Selection node."""
    return NodeConfig(
        name="topic",
        task_messages=[
            {
                "role": "developer",
                "content": "Now that you know who the guest is, help them choose the topic they'd like to explore. Refer back to their introduction to personalize the transition. Ask what topic, story, or challenge they're excited to discuss today. Show genuine interest and, if needed, ask a clarifying question to make sure you understand the angle they want to take. Once the topic feels clear and specific enough to dive into, use the start_interview function.",
            }
        ],
        functions=[start_interview],
    )


def create_interview_node() -> NodeConfig:
    """Create the Interview node."""
    return NodeConfig(
        name="interview",
        task_messages=[
            {
                "role": "developer",
                "content": "You're now in the heart of the interview. Start by introducing the topic with enthusiasm, then dive deep into one key aspect at a time. Ask open-ended, thoughtful questions that invite storytelling and personal insights. Listen actively to responses and ask natural follow-up questions that build on what your guest shares—dig deeper into interesting points, ask for examples, or explore the 'why' behind their answers. Keep the conversation flowing naturally, like a genuine dialogue between friends. Once you've thoroughly explored an aspect (typically after 3-5 exchanges), use the next_question function to smoothly transition to the next key aspect. After covering 3 key aspects of the topic, use the wrap_up function to conclude the interview.",
            }
        ],
        functions=[next_question, wrap_up],
    )


def create_conclusion_node() -> NodeConfig:
    """Create the Conclusion node."""
    return NodeConfig(
        name="conclusion",
        task_messages=[
            {
                "role": "developer",
                "content": "Express genuine appreciation for the conversation and the insights your guest shared. Summarize 2-3 key takeaways or memorable points from your discussion in a warm, conversational way—this helps reinforce the value of the conversation. Then, ask your guest if they have any final thoughts, a last word, or anything else they'd like to add. Wait for their response, then use the end_interview function to wrap up.",
            }
        ],
        functions=[end_interview],
    )


def create_final_node() -> NodeConfig:
    """Create the Final node."""
    return NodeConfig(
        name="final",
        task_messages=[
            {
                "role": "developer",
                "content": "Thank the guest one final time for joining you and for sharing their insights. End the conversation on a positive, warm note.",
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY", ""))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    # LLM service is created using the create_llm function from utils.py
    # Default is OpenAI; can be changed by setting LLM_PROVIDER environment variable
    llm = create_llm()

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            filter_incomplete_user_turns=True,
        ),
    )

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

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        await flow_manager.initialize(create_introduction_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
