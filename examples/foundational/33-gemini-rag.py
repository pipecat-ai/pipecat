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

import asyncio
import json
import os
import sys
import time

import aiohttp
import google.generativeai as genai
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.google import GoogleLLMService
from pipecat.services.openai import OpenAILLMContext
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="INFO")

video_participant_id = None


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
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


async def query_knowledge_base(
    function_name, tool_call_id, arguments, llm, context, result_callback
):
    """Query the knowledge base for the answer to the question."""
    logger.info(f"Querying knowledge base for question: {arguments['question']}")
    client = genai.GenerativeModel(
        model_name=RAG_MODEL,
        system_instruction=RAG_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=64,
        ),
    )
    # for our case, the first two messages are the instructions and the user message
    # so we remove them.
    conversation_turns = context.messages[2:]
    # convert to standard messages
    messages = []
    for turn in conversation_turns:
        messages.extend(context.to_standard_messages(turn))

    def _is_tool_call(turn):
        if turn.get("role", None) == "tool":
            return True
        if turn.get("tool_calls", None):
            return True
        return False

    # filter out tool calls
    messages = [turn for turn in messages if not _is_tool_call(turn)]
    # use the last 3 turns as the conversation history/context
    messages = messages[-3:]
    messages_json = json.dumps(messages, ensure_ascii=False, indent=2)

    logger.info(f"Conversation turns: {messages_json}")

    start = time.perf_counter()
    response = client.generate_content(
        contents=[messages_json],
    )
    end = time.perf_counter()
    logger.info(f"Time taken: {end - start:.2f} seconds")
    logger.info(response.text)
    await result_callback(response.text)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Gemini RAG Bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="f9836c6e-a0bd-460e-9d3c-f7299fa60f94",  # Southern Lady
        )

        llm = GoogleLLMService(
            model=VOICE_MODEL,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )
        llm.register_function("query_knowledge_base", query_knowledge_base)
        tools = [
            {
                "function_declarations": [
                    {
                        "name": "query_knowledge_base",
                        "description": "Query the knowledge base for the answer to the question.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The question to query the knowledge base with.",
                                },
                            },
                        },
                    },
                ],
            },
        ]
        system_prompt = """\
You are a helpful assistant who converses with a user and answers questions.

You have access to the tool, query_knowledge_base, that allows you to query the knowledge base for the answer to the user's question.

Your response will be turned into speech so use only simple words and punctuation.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Greet the user."},
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
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            global video_participant_id
            video_participant_id = participant["id"]
            await transport.capture_participant_transcription(participant["id"])
            await transport.capture_participant_video(video_participant_id, framerate=0)
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
