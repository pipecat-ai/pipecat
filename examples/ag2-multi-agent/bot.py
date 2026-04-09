#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AG2 Multi-Agent Voice Bot.

A voice bot that uses AG2 multi-agent GroupChat as the reasoning engine.
User speech is transcribed via Deepgram STT, routed through a Research Agent
and Analyst Agent coordinated by AG2's GroupChat, and the final response is
spoken back via Cartesia TTS.
"""

import asyncio
import os

from autogen import AssistantAgent, GroupChat, GroupChatManager, LLMConfig, UserProxyAgent
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import Frame, TTSSpeakFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


# --- AG2 setup ---

llm_config = LLMConfig(
    {
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_type": "openai",
    }
)


def is_termination(msg):
    content = msg.get("content", "") or ""
    return "TERMINATE" in content


proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config=False,
    is_termination_msg=is_termination,
)

research = AssistantAgent(
    name="research_agent",
    system_message=(
        "You are a research agent in a voice assistant. The message you receive is a "
        "question from a real person speaking into a microphone. Gather relevant "
        "information to answer their question. Be concise and factual. "
        "Do NOT discuss the system architecture or agent names."
    ),
    llm_config=llm_config,
)

analyst = AssistantAgent(
    name="analyst_agent",
    system_message=(
        "You synthesize research into clear, conversational answers suitable for voice "
        "output. Keep answers under 3 sentences — the user is LISTENING, not reading. "
        "Answer the user's original question directly. "
        "Do NOT mention agent names, tools, or system internals. "
        "When you have a final answer, say TERMINATE at the end."
    ),
    llm_config=llm_config,
)


@proxy.register_for_execution()
@research.register_for_llm(description="Search the web for current information on a topic")
def web_search(query: str) -> str:
    """Search the web using Tavily API. Set TAVILY_API_KEY to enable."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return f"Web search unavailable (no TAVILY_API_KEY). Topic: {query}"
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        results = client.search(query, max_results=3)
        snippets = [r.get("content", "") for r in results.get("results", [])]
        return "\n".join(snippets[:3]) if snippets else f"No results found for: {query}"
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return f"Search failed for: {query}"


# --- Pipecat FrameProcessor bridge ---


class AG2MultiAgentProcessor(FrameProcessor):
    """Bridge between pipecat's frame pipeline and AG2's multi-agent GroupChat.

    Receives TranscriptionFrame (user speech text), runs the AG2 GroupChat in a
    background thread, and pushes the final response as a TextFrame to TTS.
    """

    def __init__(self):
        super().__init__(name="AG2MultiAgentProcessor")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            logger.info(f"AG2 processing: {frame.text}")
            response = await asyncio.to_thread(self._run_ag2, frame.text)
            if response:
                await self.push_frame(TTSSpeakFrame(text=response), direction)
        else:
            await self.push_frame(frame, direction)

    def _run_ag2(self, user_text: str) -> str:
        """Run AG2 GroupChat synchronously (called via asyncio.to_thread)."""
        # Create a fresh GroupChat each turn so message history doesn't accumulate.
        # round_robin ensures: user_proxy → research_agent → analyst_agent
        group_chat = GroupChat(
            agents=[proxy, research, analyst],
            messages=[],
            max_round=10,
            speaker_selection_method="round_robin",
        )
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=llm_config,
            is_termination_msg=is_termination,
        )

        run_response = proxy.run(manager, message=user_text, silent=True)
        run_response.process()

        # Extract final response — use summary first, fall back to last message
        summary = run_response.summary
        if summary and summary != "None":
            cleaned = self._clean_response(summary)
            if cleaned:
                return cleaned

        # Walk messages in reverse to find the last assistant content
        for msg in reversed(list(run_response.messages)):
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if content and content.strip():
                cleaned = self._clean_response(content)
                if cleaned:
                    return cleaned

        logger.warning("AG2 returned no usable response")
        return "I'm sorry, I couldn't process that. Could you try again?"

    @staticmethod
    def _clean_response(text: str) -> str:
        """Strip TERMINATE marker from the response text."""
        return text.replace("TERMINATE", "").strip()


# --- Pipecat pipeline ---

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
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
    logger.info("Starting AG2 multi-agent voice bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    ag2_processor = AG2MultiAgentProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            ag2_processor,
            tts,
            transport.output(),
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
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
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
