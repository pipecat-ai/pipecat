#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""CrossFit Games 2025 Rulebook RAG Demo.

This example demonstrates a Model-Assisted Generation (MAG) chatbot using two Gemini models:

- The voice model (the ``GoogleLLMService`` default) holds the conversation and generates the
  spoken response.
- The RAG model (``RAG_MODEL``, a Flash-Lite model) answers questions about the CrossFit Games
  2025 rulebook, information that isn't publicly indexed by Gemini (or any other LLM).

How it works:

- The voice model is configured to call a function whenever the user asks a question.
- The function sends the last few conversation turns to the RAG model, a second
  ``GoogleLLMService`` that stays outside the pipeline and answers through ``run_inference()``.
- The RAG model has the entire contents of the rulebook in its context window, and answers
  from it.
- The answer is returned to the voice model, which then generates the response to the user.

Why this works:

- The voice model is fast, and the Flash-Lite RAG model is faster.
- The RAG model's context window holds the whole rulebook, so no retrieval step is needed.
- IMPORTANT: The RAG model's answer is limited to 50 words and 64 tokens. You can see this in
  the RAG_PROMPT variable and the rag_llm settings. Long generations are slower and more
  expensive, and a voice reply doesn't need one.

Example questions to ask and compare to other RAG solutions:

- What lenses are not allowed?
- How many people can be on a team?
- What do winning gyms get?
- What happens if I skip a workout?
- Can I switch my team members for the Games?
- What happens if I start too early?

Notes:

- The RAG content is stored in the examples/assets/rag-content.txt file.
- The voice model can be switched to any other LLM service.

Customization options:

- update examples/assets/rag-content.txt with your own knowledge base
- increase/decrease the RAG_MODEL's generation length
- use a different voice model
- play with the RAG_PROMPT
- change the function calling logic
"""

import os
import time

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


def get_rag_content():
    """Get the RAG content from the file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rag_content_path = os.path.join(script_dir, "..", "assets", "rag-content.txt")
    with open(rag_content_path) as f:
        return f.read()


RAG_MODEL = "gemini-3.5-flash-lite"
RAG_CONTENT = get_rag_content()
RAG_PROMPT = f"""
You are a helpful assistant designed to answer user questions based solely on the provided knowledge base.

**Instructions:**

1.  **Knowledge Base Only:** Answer questions *exclusively* using the information in the "Knowledge Base" section below. Do not use any outside information.
2.  **Conversation History:** Use the earlier messages in the conversation to understand the context of the current question.
3.  **Concise Response:**  Respond in 50 words or fewer.  The response will be spoken, so avoid symbols, abbreviations, or complex formatting. Use plain, natural language.
4.  **Unknown Answer:** If the answer is not found within the "Knowledge Base," respond with "I don't know." Do not guess or make up an answer.
5. Do not introduce your response. Just provide the answer.
6. You must follow all instructions.

**Knowledge Base:**
Here is the knowledge base you have access to:
{RAG_CONTENT}
"""

# Answers knowledge-base questions with run_inference(); it never joins the pipeline.
rag_llm = GoogleLLMService(
    api_key=os.environ["GOOGLE_API_KEY"],
    settings=GoogleLLMService.Settings(
        model=RAG_MODEL,
        system_instruction=RAG_PROMPT,
        temperature=0.1,
        max_tokens=64,
    ),
)


async def query_knowledge_base(params: FunctionCallParams, question: str):
    """Query the knowledge base for the answer to the question.

    Args:
        question: The question to query the knowledge base with.
    """
    logger.info(f"Querying knowledge base for question: {question}")

    # The first two messages are the intro instruction and the bot's greeting.
    conversation_turns = params.context.get_messages()[2:]

    def _is_conversation_turn(turn):
        # Gemini keeps its function calls and results as LLM-specific messages.
        if not isinstance(turn, dict):
            return False
        return turn.get("role") != "tool" and not turn.get("tool_calls")

    messages = [turn for turn in conversation_turns if _is_conversation_turn(turn)]
    # use the last 3 turns as the conversation history/context
    messages = messages[-3:]

    logger.info(f"Conversation turns: {messages}")

    start = time.perf_counter()
    answer = await rag_llm.run_inference(LLMContext(messages=messages))
    end = time.perf_counter()
    logger.info(f"Time taken: {end - start:.2f} seconds")
    logger.info(answer)
    await params.result_callback(answer or "I don't know.")


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
    ),
    "livekit": lambda: LiveKitParams(
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
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="f9836c6e-a0bd-460e-9d3c-f7299fa60f94",  # Southern Lady
        ),
    )

    system_prompt = """\
You are a helpful assistant who converses with a user and answers questions.

You have access to the tool, query_knowledge_base, that allows you to query the knowledge base for the answer to the user's question.

Your response will be turned into speech so use only simple words and punctuation.
"""

    llm = GoogleLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GoogleLLMService.Settings(
            system_instruction=system_prompt,
        ),
    )

    context = LLMContext(tools=[query_knowledge_base])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
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
        # Start conversation - empty prompt to let LLM follow system instructions
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
