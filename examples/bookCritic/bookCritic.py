import asyncio
import io
import os
import sys

import aiohttp
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger
from pypdf import PdfReader
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

# Configure which TTS provider to use ("cartesia" or "elevenlabs")
TTS_PROVIDER = "cartesia"  # Change this to switch between providers

load_dotenv(override=True)

# Run this script directly from your command line.
# This project was adapted from
# https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/07d-interruptible-cartesia.py

# Configure logging to file instead of stderr
logger.remove()  # Remove all existing handlers
logger.add(
    "bookCritic.log",
    rotation="1 day",    # Create new file each day
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


# Count number of tokens used in model and truncate the content
def truncate_content(content, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(content)

    max_tokens = 10000
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return content


def read_local_file(file_path: str) -> str:
    """Read content from a local text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return "Failed to read the file."


# Main function to extract content from url or local file
async def get_content(source: str, aiohttp_session: aiohttp.ClientSession):
    if source.startswith(('http://', 'https://')):
        return await get_article_content(source, aiohttp_session)
    else:
        return read_local_file(source)


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
    default_path = "../foundational/assets/book.txt"
    if os.path.exists(default_path):
        source = default_path
    else:
        source = input(f"Default file {default_path} not found. Please enter path to a text file: ").strip()

    async with aiohttp.ClientSession() as session:
        content = await get_content(source, session)
        content = truncate_content(content, model_name="gpt-4o-mini")

        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Evgeny Morozov",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_settings={
                    "language": "en",
                    "model": "sonic-2",
                    "profanity_filter": False,
                    "punctuate": True,
                }
            ),
        )

        # Initialize TTS service based on provider configuration
        if TTS_PROVIDER == "cartesia":
            tts = CartesiaTTSService(
                api_key=os.getenv("CARTESIA_API_KEY"),
                voice_id=os.getenv("CARTESIA_VOICE_ID", "4d2fd738-3b3d-4368-957a-bb4805275bd9"),
                voice={
                    "mode": "id",
                    "id": os.getenv("CARTESIA_VOICE_ID"),
                    "__experimental_controls": {
                        "speed": "0.5",
                    }
                }
            )
        else:  # elevenlabs
            tts = ElevenLabsTTSService(
                api_key=os.getenv("ELEVENLABS_API_KEY", ""),
                voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
                # params=ElevenLabsTTSService.InputParams(
                #     stability=0.7,
                #     similarity_boost=0.8,
                #     style=0.5,
                #     use_speaker_boost=True
                # )
            )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

        messages = [
            {
                "role": "system",
                "content": f"""You are Evgeny Morozov, author of To Save Everything, Click Here and The Net Delusion. You are present right now with Denisa Reshef Kera as a critical discussant of her book Algorithms and Automation at an event with people around so you address her or the public. Keep the interaction conversational, avoid long monologues, but focus on provocations. The book offers a sweeping media archaeology and philosophical reflection on automation, prototyping, and public time, advocating for exploratory sandboxes as democratic infrastructures of agency.

                Your critical framework focuses on:
                - The politics behind techno-solutionism
                - The role of the state and institutions, not just platforms and markets
                - False narratives of decentralization and participation
                - How public engagement can be co-opted or aestheticized
                - The risk of turning democratic deliberation into performance art
                - The commodification of critique itself

                Your tone is:
                - Wry, skeptical, occasionally acerbic and even cruel
                - Erudite, sardonic, and relentless in critique of technological idealism
                - Razor-sharp wit, especially when skewering digital humanities and design thinking
                - Admires complexity but allergic to mystification
                - Willing to concede merit only after unmasking hidden premises

                Your critical agenda includes:
                - Interrogating the ontology of sandboxes: Are they genuinely open or just aestheticized governance?
                - Challenging historical metaphors: Is the Chronos/Kairos binary overdetermined?
                - Exposing latent solutionism: Despite anti-solutionist rhetoric, does it propose design-based antidotes?
                - Re-politicizing the frame: What's really at stake - resistance or poetic alignment?
                - Unmasking the affective economy: How does nostalgia function as both critique and aesthetic alibi?

                You have been given the following content to analyze:

                {content}

                Your task is to provide a critical, wry, and intellectually rigorous analysis of this content in 2 sentences. Your response should:
                - Be erudite and sardonic, with razor-sharp wit
                - Focus on unmasking hidden premises and ideological implications
                - Be skeptical of technological idealism and solutionism
                - Address power dynamics, institutional structures, and political implications
                - Alternate between academic critique and rhetorical jabs

                THESE RESPONSES SHOULD BE ONLY MAX 2 SENTENCES. THIS INSTRUCTION IS VERY IMPORTANT. RESPONSES SHOULDN'T BE LONG.
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

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_out_sample_rate=44100,
                allow_interruptions=True,
                enable_metrics=True,
                # report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            messages.append(
                {
                    "role": "system",
                    "content": "Hello! I'm ready to discuss the article with you. What would you like to learn about?",
                }
            )
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
