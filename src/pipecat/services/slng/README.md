# SLNG Service

SLNG is a unified voice AI gateway that routes to multiple STT/TTS providers (Deepgram, ElevenLabs, Rime, Sarvam, and more) through a single API key and a single set of parameters. Swap the `model` string to switch providers — no other code changes needed.

## Installation

```bash
uv add "pipecat-ai[slng]"
```

## Environment variables

```env
SLNG_API_KEY=your_slng_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Model variants

Pass any SLNG model variant string to `model=`. Examples:

| Provider | STT model | TTS model |
|----------|-----------|-----------|
| Deepgram | `slng/deepgram/nova:3-en` | `slng/deepgram/aura:2-en` |
| ElevenLabs | `slng/elevenlabs/scribe:v1-en` | `slng/elevenlabs/multilingual:v2` |
| Rime | — | `slng/rime/arcana:v2` |
| Sarvam | `slng/sarvam/saarika:v2-hi` | `slng/sarvam/bulbul:v2-hi` |

## Example bot

Drop-in replacement for the Deepgram STT + TTS quickstart. Copy this to `bot.py` and run with `uv run bot.py`.

```python
#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SLNG Voice Agent

This bot uses a cascade pipeline: Speech-to-Text → LLM → Text-to-Speech
with SLNG as the unified STT and TTS gateway.

Required AI services:
- SLNG (Speech-to-Text and Text-to-Speech)
- OpenAI (LLM)

Run the bot using::

    uv run bot.py
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
from pipecat.runner.types import DailyRunnerArguments, RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.services.slng.stt import SlngSTTService
from pipecat.services.slng.tts import SlngTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv(override=True)


async def run_bot(transport: BaseTransport):
    """Main bot logic."""
    logger.info("Starting bot")

    # Speech-to-Text — swap model= to change provider with no other changes
    stt = SlngSTTService(
        api_key=os.getenv("SLNG_API_KEY"),
        model="slng/deepgram/nova:3-en",
    )

    # Text-to-Speech — swap model= and voice= to change provider
    tts = SlngTTSService(
        api_key=os.getenv("SLNG_API_KEY"),
        model="slng/deepgram/aura:2-en",
        voice="aura-2-thalia-en",
    )

    # LLM service
    llm = OpenAIResponsesLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAIResponsesLLMService.Settings(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            system_instruction=(
                "You are a helpful assistant in a voice conversation. "
                "Your responses will be spoken aloud, so avoid emojis, bullet points, "
                "or other formatting that can't be spoken. "
                "Respond to what the user said in a creative, helpful, and brief way."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[],
    )

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        context.add_message({"role": "user", "content": "Please introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
    transport = None

    match runner_args:
        case DailyRunnerArguments():
            transport = DailyTransport(
                runner_args.room_url,
                runner_args.token,
                "Pipecat Bot",
                params=DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            )
        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection
            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            )
        case _:
            logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
            return

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
```

## Region routing

Both WebSocket and HTTP services support gateway region routing via `region_override` and `world_part_override`. These map to the `X-Region-Override` and `X-World-Part-Override` request headers respectively. When both are set, `region_override` takes precedence.

Pin to a specific datacenter with `region_override`:

```python
stt = SlngSTTService(
    api_key=os.getenv("SLNG_API_KEY"),
    model="slng/deepgram/nova:3-en",
    region_override="eu-north-1",  # "ap-southeast-2" | "eu-north-1" | "us-east-1"
)
```

Or constrain to a broad geographic zone with `world_part_override`:

```python
tts = SlngTTSService(
    api_key=os.getenv("SLNG_API_KEY"),
    model="slng/deepgram/aura:2-en",
    voice="aura-2-thalia-en",
    world_part_override="eu",  # "ap" | "eu" | "na"
)
```

Both options are available on all four service classes: `SlngSTTService`, `SlngHttpSTTService`, `SlngTTSService`, `SlngHttpTTSService`.

## HTTP variants

`SlngHttpSTTService` and `SlngHttpTTSService` use the voiceai-sdk over plain HTTP instead of a persistent WebSocket. They have the same constructor signature — swap the class name and everything else stays the same.

Use the HTTP variants when:
- You want simpler infrastructure (no long-lived WS connections)
- You are doing batch transcription rather than real-time streaming

```python
from pipecat.services.slng.stt import SlngHttpSTTService
from pipecat.services.slng.tts import SlngHttpTTSService

stt = SlngHttpSTTService(api_key=os.getenv("SLNG_API_KEY"), model="slng/deepgram/nova:3-en")
tts = SlngHttpTTSService(api_key=os.getenv("SLNG_API_KEY"), model="slng/deepgram/aura:2-en", voice="aura-2-thalia-en")
```
