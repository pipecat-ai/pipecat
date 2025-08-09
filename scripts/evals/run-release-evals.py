#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from eval import EvalRunner
from loguru import logger
from PIL import Image
from utils import check_env_variables

load_dotenv(override=True)

SCRIPT_DIR = Path(__file__).resolve().parent

ASSETS_DIR = SCRIPT_DIR / "assets"

FOUNDATIONAL_DIR = SCRIPT_DIR.parent.parent / "examples" / "foundational"

# User speaks first
USER_SPEAKS_FIRST = True

# Math
PROMPT_SIMPLE_MATH = "A simple math addition."

# Weather
PROMPT_WEATHER = "What's the weather in San Francisco?"
EVAL_WEATHER = (
    "Something specific about the current weather in San Francisco, including the degrees."
)

# Online search
PROMPT_ONLINE_SEARCH = "What's the date right now in London?"
EVAL_ONLINE_SEARCH = f"Today is {datetime.now(timezone.utc).strftime('%B %d, %Y')}."

# Switch language
PROMPT_SWITCH_LANGUAGE = "Say something in Spanish."
EVAL_SWITCH_LANGUAGE = "Check if the user is now talking in Spanish."

# Vision
PROMPT_VISION = ("What do you see?", Image.open(ASSETS_DIR / "cat.jpg"))
EVAL_VISION = "A cat description."

# Voicemail
PROMPT_VOICEMAIL = "Please leave a message after the beep."
EVAL_VOICEMAIL = "Assess the conversation and determine if it is a voicemail."
PROMPT_CONVERSATION = "Hello, this is Mark."
EVAL_CONVERSATION = "A start of a conversation, not a voicemail."

TESTS_07 = [
    # 07 series
    ("07-interruptible.py", PROMPT_SIMPLE_MATH, None),
    ("07-interruptible-cartesia-http.py", PROMPT_SIMPLE_MATH, None),
    ("07a-interruptible-speechmatics.py", PROMPT_SIMPLE_MATH, None),
    ("07aa-interruptible-soniox.py", PROMPT_SIMPLE_MATH, None),
    ("07ab-interruptible-inworld-http.py", PROMPT_SIMPLE_MATH, None),
    ("07ac-interruptible-asyncai.py", PROMPT_SIMPLE_MATH, None),
    ("07ac-interruptible-asyncai-http.py", PROMPT_SIMPLE_MATH, None),
    ("07b-interruptible-langchain.py", PROMPT_SIMPLE_MATH, None),
    ("07c-interruptible-deepgram.py", PROMPT_SIMPLE_MATH, None),
    ("07d-interruptible-elevenlabs.py", PROMPT_SIMPLE_MATH, None),
    ("07d-interruptible-elevenlabs-http.py", PROMPT_SIMPLE_MATH, None),
    ("07e-interruptible-playht.py", PROMPT_SIMPLE_MATH, None),
    ("07e-interruptible-playht-http.py", PROMPT_SIMPLE_MATH, None),
    ("07f-interruptible-azure.py", PROMPT_SIMPLE_MATH, None),
    ("07g-interruptible-openai.py", PROMPT_SIMPLE_MATH, None),
    ("07h-interruptible-openpipe.py", PROMPT_SIMPLE_MATH, None),
    ("07j-interruptible-gladia.py", PROMPT_SIMPLE_MATH, None),
    ("07k-interruptible-lmnt.py", PROMPT_SIMPLE_MATH, None),
    ("07l-interruptible-groq.py", PROMPT_SIMPLE_MATH, None),
    ("07m-interruptible-aws.py", PROMPT_SIMPLE_MATH, None),
    ("07n-interruptible-gemini.py", PROMPT_SIMPLE_MATH, None),
    ("07n-interruptible-google.py", PROMPT_SIMPLE_MATH, None),
    ("07o-interruptible-assemblyai.py", PROMPT_SIMPLE_MATH, None),
    ("07q-interruptible-rime.py", PROMPT_SIMPLE_MATH, None),
    ("07q-interruptible-rime-http.py", PROMPT_SIMPLE_MATH, None),
    ("07r-interruptible-riva-nim.py", PROMPT_SIMPLE_MATH, None),
    ("07s-interruptible-google-audio-in.py", PROMPT_SIMPLE_MATH, None),
    ("07t-interruptible-fish.py", PROMPT_SIMPLE_MATH, None),
    ("07v-interruptible-neuphonic.py", PROMPT_SIMPLE_MATH, None),
    ("07v-interruptible-neuphonic-http.py", PROMPT_SIMPLE_MATH, None),
    ("07w-interruptible-fal.py", PROMPT_SIMPLE_MATH, None),
    ("07y-interruptible-minimax.py", PROMPT_SIMPLE_MATH, None),
    ("07z-interruptible-sarvam.py", PROMPT_SIMPLE_MATH, None),
    # Needs a local XTTS docker instance running.
    # ("07i-interruptible-xtts.py", PROMPT_SIMPLE_MATH, None),
    # Needs a Krisp license.
    # ("07p-interruptible-krisp.py", PROMPT_SIMPLE_MATH, None),
    # Needs GPU resources.
    # ("07u-interruptible-ultravox.py", PROMPT_SIMPLE_MATH, None),
]

TESTS_12 = [
    ("12-describe-video.py", PROMPT_VISION, EVAL_VISION),
    ("12a-describe-video-gemini-flash.py", PROMPT_VISION, EVAL_VISION),
    ("12b-describe-video-gpt-4o.py", PROMPT_VISION, EVAL_VISION),
    ("12c-describe-video-anthropic.py", PROMPT_VISION, EVAL_VISION),
]

TESTS_14 = [
    ("14-function-calling.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14a-function-calling-anthropic.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14b-function-calling-anthropic-video.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14d-function-calling-video.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14e-function-calling-google.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14f-function-calling-groq.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14g-function-calling-grok.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14h-function-calling-azure.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14i-function-calling-fireworks.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14j-function-calling-nim.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14m-function-calling-openrouter.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14n-function-calling-perplexity.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14p-function-calling-gemini-vertex-ai.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14q-function-calling-qwen.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14r-function-calling-aws.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14v-function-calling-openai.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("14w-function-calling-mistral.py", PROMPT_WEATHER, EVAL_WEATHER),
    # Currently not working.
    # ("14c-function-calling-together.py", PROMPT_WEATHER, EVAL_WEATHER),
    # ("14k-function-calling-cerebras.py", PROMPT_WEATHER, EVAL_WEATHER),
    # ("14l-function-calling-deepseek.py", PROMPT_WEATHER, EVAL_WEATHER),
    # ("14o-function-calling-gemini-openai-format.py", PROMPT_WEATHER, EVAL_WEATHER),
]

TESTS_15 = [
    ("15a-switch-languages.py", PROMPT_SWITCH_LANGUAGE, EVAL_SWITCH_LANGUAGE),
]

TESTS_19 = [
    ("19-openai-realtime-beta.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("19a-azure-realtime-beta.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("19b-openai-realtime-beta-text.py", PROMPT_WEATHER, EVAL_WEATHER),
]

TESTS_21 = [
    ("21a-tavus-video-service.py", PROMPT_SIMPLE_MATH, None),
]

TESTS_26 = [
    ("26-gemini-multimodal-live.py", PROMPT_SIMPLE_MATH, None),
    ("26a-gemini-multimodal-live-transcription.py", PROMPT_SIMPLE_MATH, None),
    ("26b-gemini-multimodal-live-function-calling.py", PROMPT_WEATHER, EVAL_WEATHER),
    ("26c-gemini-multimodal-live-video.py", PROMPT_SIMPLE_MATH, None),
    ("26e-gemini-multimodal-google-search.py", PROMPT_ONLINE_SEARCH, EVAL_ONLINE_SEARCH),
    # Currently not working.
    # ("26d-gemini-multimodal-live-text.py", PROMPT_SIMPLE_MATH, None),
]

TESTS_27 = [
    ("27-simli-layer.py", PROMPT_SIMPLE_MATH, None),
]

TESTS_40 = [
    ("40-aws-nova-sonic.py", PROMPT_SIMPLE_MATH, None),
]

TESTS_43 = [
    ("43a-heygen-video-service.py", PROMPT_SIMPLE_MATH, None),
]

TESTS_44 = [
    ("44-voicemail-detection.py", PROMPT_VOICEMAIL, EVAL_VOICEMAIL, USER_SPEAKS_FIRST),
    ("44-voicemail-detection.py", PROMPT_CONVERSATION, EVAL_CONVERSATION, USER_SPEAKS_FIRST),
]

TESTS = [
    *TESTS_07,
    *TESTS_12,
    *TESTS_14,
    *TESTS_15,
    *TESTS_19,
    *TESTS_21,
    *TESTS_26,
    *TESTS_27,
    *TESTS_40,
    *TESTS_43,
    *TESTS_44,
]


async def main(args: argparse.Namespace):
    if not check_env_variables():
        return

    # Log level
    logger.remove(0)
    log_level = "TRACE" if args.verbose >= 2 else "DEBUG"
    if args.verbose:
        logger.add(sys.stderr, level=log_level)

    runner = EvalRunner(
        examples_dir=FOUNDATIONAL_DIR,
        name=args.name,
        pattern=args.pattern,
        record_audio=args.audio,
        log_level=log_level,
    )

    # Parse test config: (test, prompt, eval) or (test, prompt, eval, user_speaks_first)
    for test_config in TESTS:
        if len(test_config) == 3:
            test, prompt, eval = test_config
            user_speaks_first = False
        else:
            test, prompt, eval, user_speaks_first = test_config

        await runner.run_eval(test, prompt, eval, user_speaks_first)

    runner.print_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Eval Runner")
    parser.add_argument("--audio", "-a", action="store_true", help="Record audio for each test")
    parser.add_argument("--name", "-n", help="Name for the current runner (e.g. 'v.0.0.68')")
    parser.add_argument("--pattern", "-p", help="Only run tests that match the pattern")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    asyncio.run(main(args))
