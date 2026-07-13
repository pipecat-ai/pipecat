#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice formatting with individual text transforms.

Demonstrates how to compose individual built-in text transforms instead of
using the VoiceFormatter bundle, giving fine-grained control over which
transforms are applied and in what order.

Each transform is an async callable with the signature:
    async def transform(text: str, aggregation_type: str) -> str

Transforms are registered via text_transforms on the TTS service as a list of
(aggregation_type, callable) pairs. The aggregation_type string controls which
frames the transform applies to ("*" means all frames).

This example shows a billing-assistant scenario where several transforms are
composed:
  - strip_markdown       Remove bold/italic/headers the LLM might add
  - normalize_acronyms   "API" → "A P I"
  - email_to_speech      "user@example.com" → "user at example dot com"
  - expand_currency      "$42.50" → "forty-two dollars and fifty cents"
  - expand_percentages   "3.5%" → "three point five percent"
  - replace_text         Custom substitutions (e.g. "Dr." → "Doctor"), including
                          an SSML phoneme tag for a word ElevenLabs would
                          otherwise mispronounce.

Run locally:
    python features-text-transforms.py

Run against a Daily room:
    python features-text-transforms.py -t daily

Requires:
    pip install pipecat-ai[cartesia,deepgram,openai,silero,daily]
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.utils.text.transforms import (
    email_to_speech,
    expand_currency,
    expand_percentages,
    normalize_acronyms,
    replace_text,
    strip_markdown,
)
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

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

    stt = CartesiaSTTService(api_key=os.environ["CARTESIA_API_KEY"])

    # Custom substitution rules applied after all other transforms.
    # Patterns are regular expressions; use re.escape() for literal strings.
    #
    # The last rule wraps a name in an SSML phoneme tag so ElevenLabs
    # pronounces it correctly.
    custom_subs = replace_text(
        [
            (r"\bDr\.", "Doctor"),
            (r"\bSt\.", "Street"),
            (r"\bApt\.", "Apartment"),
            (r"\bvs\b", "versus"),
            # IPA phoneme tags are only supported on ElevenLabs v2 models, and you need to set enable_ssml_parsing=True.
            # More details here: https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices#phoneme-tags-for-v2-models
            # (r"(?i)\bSiobhan\b", '<phoneme alphabet="ipa" ph="ʃəˈvɔːn">Siobhan</phoneme>'),
            # This is an alternative that works on all models.
            (r"(?i)\bSiobhan\b", "shi-VAWN"),
        ]
    )

    # Build a transform chain for a billing-assistant use case.
    # Order matters: strip markdown first, then expand special patterns.
    billing_transforms = [
        ("*", strip_markdown),
        ("*", normalize_acronyms),
        ("*", email_to_speech),
        ("*", expand_currency),
        ("*", expand_percentages),
        ("*", custom_subs),
    ]

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID", ""),
            # Set a v2 model when using IPA phoneme tags.
            # model="eleven_flash_v2"
        ),
        # Enable SSML parsing for ElevenLabs v2 models.
        # enable_ssml_parsing=True,
        text_transforms=billing_transforms,
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are Siobhan, a billing support assistant for Nexora, a telecom "
                "company. Your responses are spoken aloud. Use natural formatting in "
                "your answers: currency amounts like $42.50, percentages like 3.5%, "
                "email addresses like support@example.com, and abbreviations like "
                "Dr. or St. as you normally would in writing — the voice formatter "
                "will convert them to natural speech before synthesis."
            ),
        ),
    )

    context = LLMContext()
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Introduce yourself by name and let the caller know they've "
                    "reached billing support. Offer to help with their account "
                    "balance, recent charges, or payment options. Give a sample "
                    "balance such as $127.50 due on 3/15/2025 and a support email "
                    "like billing@telecom.example.com."
                ),
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
