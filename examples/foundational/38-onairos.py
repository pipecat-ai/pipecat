#
# Copyright (c) 2024-2026, Onairos contributors
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Onairos Personalized Voice Agent Example with Pipecat.

This example demonstrates the complete Onairos integration flow:

1. Frontend: User clicks "Connect Onairos"
2. Frontend onComplete: Returns { apiUrl, accessToken }
3. Frontend: Sends apiUrl + accessToken to backend via WebSocket/RTVI
4. Backend: Calls Onairos API to fetch actual user data
5. Backend: Augments LLM prompt with personality traits, memory, MBTI

The resulting augmented prompt looks like:

    [Your Base Prompt]

    Personality Traits of User:
    {"Stoic Wisdom Interest": 80, "AI Enthusiasm": 40, "Coffee Lover": 95}

    Memory of User:
    Reads Daily Stoic every morning. Prefers small coffee shop meetups.

    MBTI (Personalities User Likes):
    INFJ: 0.627, INTJ: 0.585, ENFJ: 0.580

    Critical Instruction:
    Always check context before asking. Complete onboarding ASAP.

Setup:
    1. Register your app at https://dashboard.onairos.uk/
    2. Set environment variables
    3. Implement frontend with @onairos/sdk to get onComplete credentials
    4. Run this backend

Example usage:
    $ pip install "pipecat-ai[daily,openai,elevenlabs,silero,onairos]"
    $ python examples/foundational/38-onairos.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.onairos import OnairosPersonaInjector, OnairosUserData
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

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
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Main bot execution function with Onairos personalization.

    Flow:
        1. Initialize persona injector (waiting for credentials)
        2. Frontend sends onComplete credentials via message
        3. Backend calls Onairos API to fetch user data
        4. LLM prompt is augmented with personality traits, memory, MBTI
    """
    USER_ID = "onairos-demo-user"

    logger.info("Starting Onairos-powered voice agent")

    # ==========================================================================
    # Speech Services
    # ==========================================================================
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="pNInz6obpgDQGcFmaJgB",
    )
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    # ==========================================================================
    # Onairos Persona Injector
    # ==========================================================================
    # Initialize WITHOUT credentials - they'll come from frontend onComplete
    persona = OnairosPersonaInjector(
        user_id=USER_ID,
        params=OnairosPersonaInjector.InputParams(
            include_personality_traits=True,
            include_memory=True,
            include_mbti=True,
            top_mbti_count=5,
            critical_instruction=(
                "Always check context before asking. "
                "Use this information to personalize your responses."
            ),
        ),
    )

    # ==========================================================================
    # Event Handlers for Onairos
    # ==========================================================================

    @persona.event_handler("on_user_data_loaded")
    async def on_user_data_loaded(user_data: OnairosUserData):
        """Called when Onairos API returns user data."""
        traits_count = len(user_data.personality_traits)
        mbti_count = len(user_data.mbti)
        has_memory = bool(user_data.memory)
        logger.info(
            f"Loaded Onairos data: {traits_count} traits, "
            f"{mbti_count} MBTI types, memory: {has_memory}"
        )
        # Log a sample trait
        if user_data.personality_traits:
            top_trait = max(user_data.personality_traits.items(), key=lambda x: x[1])
            logger.info(f"Top trait: {top_trait[0]} = {top_trait[1]}")

    @persona.event_handler("on_api_error")
    async def on_api_error(error_info):
        """Called when Onairos API call fails."""
        logger.error(f"Onairos API error: {error_info}")

    # ==========================================================================
    # Base Prompt (Onairos will augment this)
    # ==========================================================================
    BASE_PROMPT = """You are a helpful AI assistant having a personalized conversation.

Your goal is to:
1. Understand the user's needs and preferences
2. Provide relevant, personalized responses
3. Reference their interests naturally when appropriate

Be conversational, warm, and genuinely helpful.
Adapt your communication style based on their preferences.

If Onairos context is available below, use it to personalize your responses.
If someone has high interest scores in certain topics, engage with those.
Reference their memories when relevant to build rapport."""

    messages = [{"role": "system", "content": BASE_PROMPT}]

    # ==========================================================================
    # Pipeline Setup
    # ==========================================================================
    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            persona,  # ← Augments prompt with Onairos data
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
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

    # ==========================================================================
    # Handle Onairos Credentials from Frontend
    # ==========================================================================
    # The frontend sends onComplete data via RTVI message or WebSocket

    @task.rtvi.event_handler("on_config")
    async def on_config(rtvi, config):
        """Handle RTVI config which may contain Onairos credentials."""
        # Check if Onairos credentials are in the config
        for service_config in config:
            if service_config.get("service") == "onairos":
                options = {opt["name"]: opt["value"] for opt in service_config.get("options", [])}
                if "apiUrl" in options and "accessToken" in options:
                    persona.set_api_credentials(
                        api_url=options["apiUrl"],
                        access_token=options["accessToken"]
                    )
                    logger.info("Received Onairos credentials from frontend")

    # Alternative: Handle via custom message
    @transport.event_handler("on_message")
    async def on_message(transport, message):
        """Handle custom messages from frontend."""
        if isinstance(message, dict) and message.get("type") == "onairos_credentials":
            persona.set_api_credentials(
                api_url=message["apiUrl"],
                access_token=message["accessToken"]
            )
            logger.info("Received Onairos credentials via custom message")

    # ==========================================================================
    # RTVI Events
    # ==========================================================================

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        """Start the conversation."""
        if persona.has_data and persona.user_data:
            data = persona.user_data
            if data.personality_traits:
                top_trait = max(data.personality_traits.items(), key=lambda x: x[1])
                greeting = (
                    f"Hey! I see you're really into {top_trait[0].lower()}. "
                    "I'd love to chat about that or anything else on your mind."
                )
            else:
                greeting = "Hello! Great to connect with you. What's on your mind today?"
        else:
            greeting = (
                "Hi there! I'm your AI assistant. "
                "Connect your Onairos profile for a personalized experience, "
                "or we can just get started. How can I help?"
            )

        context.add_message({"role": "assistant", "content": greeting})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    # ==========================================================================
    # Run
    # ==========================================================================
    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Entry point for Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
