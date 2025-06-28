#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os
import re
import asyncio
from typing import List, Set
import websockets
from websockets.server import WebSocketServerProtocol
from pipecat.processors.text_transformer import StatelessTextTransformer
from typing import List

from dotenv import load_dotenv
load_dotenv(override=True)
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.stt_service import SegmentedSTTService
from lightning_whisper_mlx import LightningWhisperMLX
import numpy as np
import tempfile
import soundfile as sf
from typing import AsyncGenerator
from pipecat.frames.frames import Frame, TranscriptionFrame, ErrorFrame
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
# Added to supply extra parameters (think:false) to Ollama requests
from pipecat.services.openai.base_llm import BaseOpenAILLMService

load_dotenv(override=True)


# ---- Lightning Whisper STT integration -----------------------


class LightningWhisperSTTService(WhisperSTTService):
    """STT service that wraps lightning-whisper-mlx for fast Apple-Silicon decoding."""

    def __init__(self, model: str = "distil-medium.en", quant: str | None = None, **kwargs):
        # Skip WhisperSTTService.__init__ to avoid loading default Whisper weights
        SegmentedSTTService.__init__(self, **kwargs)
        self._model_name = model
        self._quant = quant
        self._whisper = LightningWhisperMLX(model=model, quant=quant)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:  # type: ignore
        """Override to run Lightning-Whisper on the captured audio chunk."""
        try:
            # Pipecat provides signed 16-bit PCM; convert to WAV temp file for lightning.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                # Convert to float32 for soundfile.write (normalises later but we keep int16 to save time)
                pcm = np.frombuffer(audio, dtype=np.int16)
                sf.write(tmp, pcm, samplerate=16000, subtype="PCM_16")
                tmp_path = tmp.name
            result = self._whisper.transcribe(tmp_path)
            text = result.get("text", "")
            yield TranscriptionFrame(text=text, user_id="", timestamp="", language=None)
        except Exception as e:
            yield ErrorFrame(error=str(e))


_stream_clients: Set[WebSocketServerProtocol] = set()


async def _ws_handler(ws: WebSocketServerProtocol, _path):
    _stream_clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        _stream_clients.discard(ws)


async def _broadcast(text: str):
    if not _stream_clients:
        return text
    dead = []
    for client in _stream_clients:
        try:
            await client.send(text)
        except Exception:
            dead.append(client)
    for d in dead:
        _stream_clients.discard(d)
    return text


# ---- RAG SETUP --------------------------------------------------
# Load and chunk the local knowledge base once, on import


def _load_kb() -> List[str]:
    """Load and chunk the knowledge-base file.

    Preferred format (gold-standard): Markdown with level-2 headings ("## Heading").
    Fallback: the legacy YAML-style `key: >` blocks that are currently present in
    some deployments.  The splitter auto-detects which pattern gives more than
    one chunk so you can migrate the KB incrementally – simply switch the file
    to Markdown and restart, no code changes needed.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(script_dir, "assets", "rag-content.txt")
    if not os.path.exists(kb_path):
        logger.warning(f"Knowledge base file not found: {kb_path}")
        return []

    with open(kb_path, "r", encoding="utf-8") as f:
        text = f.read()

    # --- 1. Try Markdown H2 splitter -------------------------------------------------
    md_chunks = re.split(r'\n(?=## )', text)
    md_chunks = [c.strip() for c in md_chunks if c.strip()]

    # --- 2. Fallback to YAML-like key splitter ---------------------------------------
    if len(md_chunks) <= 1:
        yaml_chunks = re.split(r'\n(?=\w+:\s*(?:$|>))', text)
        chunks = [c.strip() for c in yaml_chunks if c.strip()]
        scheme = "YAML-like"
    else:
        chunks = md_chunks
        scheme = "Markdown H2"

    logger.info(
        f"Loaded and split knowledge base into {len(chunks)} chunks using {scheme} strategy.")
    return chunks


_KB_CHUNKS: List[str] = _load_kb()

# top_k controls how many retrieved chunks are injected per turn –
# lower is faster (less context), higher may improve accuracy.
def _retrieve_chunks(query: str, kb_chunks: List[str], top_k: int = 4) -> List[str]:
    """Very simple keyword overlap ranking – fast and local."""
    if not kb_chunks or not query:
        return []
    q_words = set(re.findall(r"[\w']+", query.lower()))
    scored = []
    for chunk in kb_chunks:
        words = set(re.findall(r"[\w']+", chunk.lower()))
        score = len(q_words & words)
        if score:
            scored.append((score, chunk))
    # Fallback: if nothing matches, return first paragraph to avoid empty list
    if not scored:
        return kb_chunks[:top_k]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


class RAGOpenAILLMContext(OpenAILLMContext):
    """Injects retrieved KB chunks into the user's turn using the format expected by the Modelfile."""

    def __init__(self, kb_chunks: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kb_chunks = kb_chunks
        # Remember how many static few-shot messages were provided at startup so we can
        # prune history back to this baseline every turn (stateless RAG).
        self._static_count = len(self._messages)

    def add_message(self, message):
        # Only transform real user messages (not the few-shot examples we keep at boot time).
        # For every new turn we clear dynamic history to keep the prompt lean.
        # This prevents the messages list from growing without bound.
        if len(self._messages) > self._static_count:
            self._messages = self._messages[: self._static_count]

        if message.get("role") == "user":
            user_query = message.get("content", "")
            retrieved = _retrieve_chunks(user_query, self._kb_chunks, top_k=3)
            if retrieved:
                context_str = "\n\n---\n\n".join(retrieved)
                structured = f"""Context:\n{context_str}\n\n---\nQuestion: {user_query}"""
                message["content"] = structured
        super().add_message(message)


# -------------------------------------------------------------------

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    #   "daily": lambda: DailyParams(
    #      audio_in_enabled=True,
    #     audio_out_enabled=True,
    #    vad_analyzer=SileroVADAnalyzer(),
    #   ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    # Force a unique, high port to avoid all conflicts
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
        port=27880,
    ),
}


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"run_example started for new client: transport={transport}")

    # Move _broadcast and _KB_CHUNKS here if not immutable or if they are session-specific
    # Example placeholder:
    # def _broadcast(text):
    #     ...
    # _KB_CHUNKS = ...

    logger.info(f"Starting bot")

    stt = LightningWhisperSTTService()

    tts = OpenAITTSService(
        api_key="not-needed",
        base_url="http://localhost:8880/v1",
        voice="shimmer",
        model="kokoro",
        speed=0.8,
    )

    # Instantiate local Ollama model.
    llm = OLLamaLLMService(
        model=os.getenv("OLLAMA_MODEL", "gemma3n"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )


    messages = [

    {
        "role": "user",
        "content": "How can I contact Alex?"
    },
    {
        "role": "assistant",
        "content": "You can reach Alex by email at alex at alex covo dot com, or visit the website alex covo dot com."
    },
    {
        "role": "user",
        "content": "What's the best way to get in touch?"
    },
    {
        "role": "assistant",
        "content": "The best way to get in touch is by emailing alex at alex covo dot com."
    },
    {
        "role": "user",
        "content": "Does Alex have Instagram?"
    },
    {
        "role": "assistant",
        "content": "Yes, Alex is on Instagram. You can find him at insta at alex covo."
    },
    {
        "role": "user",
        "content": "What's his LinkedIn?"
    },
    {
        "role": "assistant",
        "content": "Alex's LinkedIn is linked in at alex covo."
    },
    {
        "role": "user",
        "content": "How do I find Alex on X?"
    },
    {
        "role": "assistant",
        "content": "On X, look for alex covo underscore eth."
    },
    {
        "role": "user",
        "content": "Has Alex done any work with AI?"
    },
    {
        "role": "assistant",
        "content": "Yes, Alex has hands-on experience with AI. In fact, he built this privacy-first, fully local conversational AI assistant that you're using now."
    },
    {
        "role": "user",
        "content": "How was this assistant built?"
    },
    {
        "role": "assistant",
        "content": "This assistant was built using open-source large language models, local speech-to-text and text-to-speech, and a modular pipeline architecture called Pipecat. All processing happens on your own device for privacy."
    },
    {
        "role": "user",
        "content": "What frameworks or tools did Alex use?"
    },
    {
        "role": "assistant",
        "content": "Alex used frameworks and tools like CrewAI, Pipecat, Ollama, LightningWhisperMLX, OpenAI TTS, and Docker to build and orchestrate this system."
    }
    ]

    context = RAGOpenAILLMContext(_KB_CHUNKS, messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            StatelessTextTransformer(_broadcast),  # stream text
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
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

    # ------------- LEAVE A MESSAGE FEATURE (CSV LOGGING) ------------- #
    import csv
    from datetime import datetime

    # Tracks clients in message mode and accumulates their message parts
    leave_message_clients = {}
    def save_message_to_csv(client_id, message, filename="user_messages.csv"):
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().isoformat(), client_id, message])

    async def handle_leave_message(client_id, user_text):
        done_keywords = ["no", "done", "that's it", "finished", "nope", "nah", "no thank you", "I'm good", "nope", "that's all", "no that's all", "nothing else"]
        # If client is in message mode, accumulate or finish
        if client_id in leave_message_clients:
            # Check if user said they're done
            if any(kw in user_text.strip().lower() for kw in done_keywords):
                message = leave_message_clients.pop(client_id)
                save_message_to_csv(client_id, message.strip())
                return "Your message has been recorded and will be passed along. Is there anything else you need?"
            else:
                leave_message_clients[client_id] += " " + user_text.strip()
                return "Is there anything else you’d like to add to your message? Say 'no' to finish."
        # Detect intent to leave a message
        elif any(phrase in user_text.lower() for phrase in ["leave a message", "leave feedback", "contact alex"]):
            leave_message_clients[client_id] = ""
            return "Sure, please dictate your message after the beep. Beeeeep."
        else:
            return None
    # ------------- END LEAVE A MESSAGE FEATURE (CSV LOGGING) ------------- #

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


# WebSocket server will be managed by FastAPI startup/shutdown events in run_multisession.py

if __name__ == "__main__":
    from pipecat.examples.run_multisession import main

    main(run_example, transport_params=transport_params)
