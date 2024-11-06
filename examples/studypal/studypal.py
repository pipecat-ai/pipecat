import aiohttp
import asyncio
import os
import sys
import io

from bs4 import BeautifulSoup
from pypdf import PdfReader
import tiktoken

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from runner import configure

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

# Run this script directly from your command line.
# This project was adapted from
# https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/07d-interruptible-cartesia.py

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# Count number of tokens used in model and truncate the content
def truncate_content(content, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(content)

    max_tokens = 10000
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return content


# Main function to extract content from url


async def get_article_content(url: str, aiohttp_session: aiohttp.ClientSession):
    if "arxiv.org" in url:
        return await get_arxiv_content(url, aiohttp_session)
    else:
        return await get_wikipedia_content(url, aiohttp_session)


# Helper function to extract content from Wikipedia url (this is
# technically agnostic to URL type but will work best with Wikipedia
# articles)


async def get_wikipedia_content(url: str, aiohttp_session: aiohttp.ClientSession):
    async with aiohttp_session.get(url) as response:
        if response.status != 200:
            return "Failed to download Wikipedia article."

        text = await response.text()
        soup = BeautifulSoup(text, "html.parser")

        content = soup.find("div", {"class": "mw-parser-output"})

        if content:
            return content.get_text()
        else:
            return "Failed to extract Wikipedia article content."


# Helper function to extract content from arXiv url


async def get_arxiv_content(url: str, aiohttp_session: aiohttp.ClientSession):
    if "/abs/" in url:
        url = url.replace("/abs/", "/pdf/")
    if not url.endswith(".pdf"):
        url += ".pdf"

    async with aiohttp_session.get(url) as response:
        if response.status != 200:
            return "Failed to download arXiv PDF."

        content = await response.read()
        pdf_file = io.BytesIO(content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text


# This is the main function that handles STT -> LLM -> TTS


async def main():
    url = input("Enter the URL of the article you would like to talk about: ")

    async with aiohttp.ClientSession() as session:
        article_content = await get_article_content(url, session)
        article_content = truncate_content(article_content, model_name="gpt-4o-mini")

        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "studypal",
            DailyParams(
                audio_out_sample_rate=44100,
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=os.getenv("CARTESIA_VOICE_ID", "4d2fd738-3b3d-4368-957a-bb4805275bd9"),
            # British Narration Lady: 4d2fd738-3b3d-4368-957a-bb4805275bd9
            params=CartesiaTTSService.InputParams(
                sample_rate=44100,
            ),
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

        messages = [
            {
                "role": "system",
                "content": f"""You are an AI study partner. You have been given the following article content:

{article_content}

Your task is to help the user understand and learn from this article in 2 sentences. THESE RESPONSES SHOULD BE ONLY MAX 2 SENTENCES. THIS INSTRUCTION IS VERY IMPORTANT. RESPONSES SHOULDN'T BE LONG.
""",
            },
        ]

        context = OpenAILLMContext(messages)
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

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True, enable_metrics=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            messages.append(
                {
                    "role": "system",
                    "content": "Hello! I'm ready to discuss the article with you. What would you like to learn about?",
                }
            )
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
