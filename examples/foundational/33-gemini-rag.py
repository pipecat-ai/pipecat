#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""CrossFit Games 2025 Rulebook RAG Demo.

This example demonstrates a Model-Assisted Generation (MAG) chatbot using Google's Gemini model.
This example uses 2 Gemini models:
- Gemini 2.0 Flash: This is the voice model that is used to generate the response.
- Gemini 2.0 Flash Lite: This is the model that is used to answer questions about the CrossFit Games 2025 rulebook - information that isn't yet publicly
indexed by Gemini (or any other LLM).

How it works:
- The voice model (Gemini 2.0 Flash) is configured to call a function whenever the user asks a question.
- The function call is a tool call to the MAG model (Gemini 2.0 Flash Lite).
- The MAG model generates a response based on the question. The MAG model has the entire contents of the CrossFit Games 2025 rulebook in it's context window.
- The response is returned to the voice model (Gemini 2.0 Flash), which then generates the response to the user.

Why this works:
- Gemini 2.0 Flash is fast
- Gemini 2.0 Flash Lite is faster
- Gemini 2.0 Flash Lite has a large (1 million tokens) context window
- IMPORTANT: The generated response from Gemini 2.0 Flash Lite is limited to 50 words or less and 64 tokens.
You can see this in the RAG_PROMPT variable and the generation_config in the query_knowledge_base function.
Long generations are slower and more expensive, in the world of Voice AI, we don't need long generations.

Example questions to ask and compare to other RAG solutions:
- What lenses are not allowed?
- How many people can be on a team?
- What do winning gyms get?
- What happens if I skip a workout?
- Can I switch my team members for the Games?
- What happens if I start too early?

Notes:
- The RAG model is Gemini 2.0 Flash Lite.
- The voice model is Gemini 2.0 Flash.
- The RAG content is stored in the assets/rag-content.txt file.
- The model for voice is Gemini 2.0 Flash, but can be easily switched to any other model.

Customization options:
- update assets/rag-content.txt with your own knowledge base
- increase/decrease the RAG_MODEL's generation length
- use a different voice model
- play with the RAG_PROMPT
- change the function calling logic
"""

import json
import os
import time

from dotenv import load_dotenv
from google import genai
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# Initialize the client globally
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def get_rag_content():
    """Get the RAG content from the file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rag_content_path = os.path.join(script_dir, "assets", "rag-content.txt")
    with open(rag_content_path, "r") as f:
        return f.read()


RAG_MODEL = "gemini-2.0-flash-lite-preview-02-05"
VOICE_MODEL = "gemini-2.0-flash"
RAG_CONTENT = get_rag_content()
RAG_PROMPT = f"""
You are a helpful assistant designed to answer user questions based solely on the provided knowledge base.

**Instructions:**

1.  **Knowledge Base Only:** Answer questions *exclusively* using the information in the "Knowledge Base" section below. Do not use any outside information.
2.  **Conversation History:** Use the "Conversation History" (ordered oldest to newest) to understand the context of the current question.
3.  **Concise Response:**  Respond in 50 words or fewer.  The response will be spoken, so avoid symbols, abbreviations, or complex formatting. Use plain, natural language.
4.  **Unknown Answer:** If the answer is not found within the "Knowledge Base," respond with "I don't know." Do not guess or make up an answer.
5. Do not introduce your response. Just provide the answer.
6. You must follow all instructions.

**Input Format:**

Each request will include:

*   **Conversation History:**  (A list of previous user and assistant messages, if any)

**Knowledge Base:**
Here is the knowledge base you have access to:
{RAG_CONTENT}
"""


async def query_knowledge_base(params: FunctionCallParams):
    """Query the knowledge base for the answer to the question."""
    logger.info(f"Querying knowledge base for question: {params.arguments['question']}")

    # for our case, the first two messages are the instructions and the user message
    # so we remove them.
    conversation_turns = params.context.get_messages()[2:]

    def _is_tool_call(turn):
        if turn.get("role", None) == "tool":
            return True
        if turn.get("tool_calls", None):
            return True
        return False

    # filter out tool calls
    messages = [turn for turn in conversation_turns if not _is_tool_call(turn)]
    # use the last 3 turns as the conversation history/context
    messages = messages[-3:]
    messages_json = json.dumps(messages, ensure_ascii=False, indent=2)

    logger.info(f"Conversation turns: {messages_json}")

    start = time.perf_counter()
    full_prompt = f"System: {RAG_PROMPT}\n\nConversation History: {messages_json}"

    response = await client.aio.models.generate_content(
        model=RAG_MODEL,
        contents=[full_prompt],
        config={
            "temperature": 0.1,
            "max_output_tokens": 64,
        },
    )
    end = time.perf_counter()
    logger.info(f"Time taken: {end - start:.2f} seconds")
    logger.info(response.text)
    await params.result_callback(response.text)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="f9836c6e-a0bd-460e-9d3c-f7299fa60f94",  # Southern Lady
    )

    llm = GoogleLLMService(
        model=VOICE_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    llm.register_function("query_knowledge_base", query_knowledge_base)

    query_function = FunctionSchema(
        name="query_knowledge_base",
        description="Query the knowledge base for the answer to the question.",
        properties={
            "question": {
                "type": "string",
                "description": "The question to query the knowledge base with.",
            },
        },
        required=["question"],
    )
    tools = ToolsSchema(standard_tools=[query_function])

    system_prompt = """\
You are a helpful assistant who converses with a user and answers questions.

You have access to the tool, query_knowledge_base, that allows you to query the knowledge base for the answer to the user's question.

Your response will be turned into speech so use only simple words and punctuation.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Greet the user."},
    ]

    context = LLMContext(messages, tools)
    context_aggregator = LLMContextAggregatorPair(context)

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
