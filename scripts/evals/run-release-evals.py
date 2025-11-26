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
from eval import EvalConfig, EvalRunner
from loguru import logger
from PIL import Image
from utils import check_env_variables

load_dotenv(override=True)

SCRIPT_DIR = Path(__file__).resolve().parent

ASSETS_DIR = SCRIPT_DIR / "assets"

FOUNDATIONAL_DIR = SCRIPT_DIR.parent.parent / "examples" / "foundational"

EVAL_SIMPLE_MATH = EvalConfig(
    prompt="A simple math addition.",
    eval="The user answers the math addition correctly.",
)

EVAL_WEATHER = EvalConfig(
    prompt="What's the weather in San Francisco (in farhenheit or celsius)?",
    eval="The user says something specific about the current weather in San Francisco, including the degrees (in farhenheit or celsius).",
)

EVAL_ONLINE_SEARCH = EvalConfig(
    prompt="What's the date right now in London?",
    eval=f"The user says today is {datetime.now(timezone.utc).strftime('%B %d, %Y')} in London.",
)

EVAL_SWITCH_LANGUAGE = EvalConfig(
    prompt="Say something in Spanish.",
    eval="The user talks in Spanish.",
)

EVAL_VISION_CAMERA = EvalConfig(
    prompt=("Briefly describe what you see.", Image.open(ASSETS_DIR / "cat.jpg")),
    eval="The user provides a cat description.",
)


def EVAL_VISION_IMAGE(*, eval_speaks_first: bool = False):
    return EvalConfig(
        prompt="Briefly describe this image.",
        eval="The user provides a cat description.",
        eval_speaks_first=eval_speaks_first,
        runner_args_body={
            "image_path": ASSETS_DIR / "cat.jpg",
            "question": "Briefly describe this image.",
        },
    )


EVAL_VOICEMAIL = EvalConfig(
    prompt="Please leave a message.",
    eval="The user leaves a voicemail message.",
    eval_speaks_first=True,
)

EVAL_CONVERSATION = EvalConfig(
    prompt="Hello, this is Mark.",
    eval="The user acknowledges the greeting.",
    eval_speaks_first=True,
)


TESTS_07 = [
    # 07 series
    ("07-interruptible.py", EVAL_SIMPLE_MATH),
    ("07-interruptible-cartesia-http.py", EVAL_SIMPLE_MATH),
    ("07a-interruptible-speechmatics.py", EVAL_SIMPLE_MATH),
    ("07aa-interruptible-soniox.py", EVAL_SIMPLE_MATH),
    ("07ab-interruptible-inworld-http.py", EVAL_SIMPLE_MATH),
    ("07ac-interruptible-asyncai.py", EVAL_SIMPLE_MATH),
    ("07ac-interruptible-asyncai-http.py", EVAL_SIMPLE_MATH),
    ("07b-interruptible-langchain.py", EVAL_SIMPLE_MATH),
    ("07c-interruptible-deepgram.py", EVAL_SIMPLE_MATH),
    ("07c-interruptible-deepgram-flux.py", EVAL_SIMPLE_MATH),
    ("07c-interruptible-deepgram-http.py", EVAL_SIMPLE_MATH),
    ("07d-interruptible-elevenlabs.py", EVAL_SIMPLE_MATH),
    ("07d-interruptible-elevenlabs-http.py", EVAL_SIMPLE_MATH),
    ("07f-interruptible-azure.py", EVAL_SIMPLE_MATH),
    ("07g-interruptible-openai.py", EVAL_SIMPLE_MATH),
    ("07h-interruptible-openpipe.py", EVAL_SIMPLE_MATH),
    ("07j-interruptible-gladia.py", EVAL_SIMPLE_MATH),
    ("07k-interruptible-lmnt.py", EVAL_SIMPLE_MATH),
    ("07l-interruptible-groq.py", EVAL_SIMPLE_MATH),
    ("07m-interruptible-aws.py", EVAL_SIMPLE_MATH),
    ("07m-interruptible-aws-strands.py", EVAL_WEATHER),
    ("07n-interruptible-gemini.py", EVAL_SIMPLE_MATH),
    ("07n-interruptible-google.py", EVAL_SIMPLE_MATH),
    ("07o-interruptible-assemblyai.py", EVAL_SIMPLE_MATH),
    ("07q-interruptible-rime.py", EVAL_SIMPLE_MATH),
    ("07q-interruptible-rime-http.py", EVAL_SIMPLE_MATH),
    ("07r-interruptible-riva-nim.py", EVAL_SIMPLE_MATH),
    ("07s-interruptible-google-audio-in.py", EVAL_SIMPLE_MATH),
    ("07t-interruptible-fish.py", EVAL_SIMPLE_MATH),
    ("07v-interruptible-neuphonic.py", EVAL_SIMPLE_MATH),
    ("07v-interruptible-neuphonic-http.py", EVAL_SIMPLE_MATH),
    ("07w-interruptible-fal.py", EVAL_SIMPLE_MATH),
    ("07y-interruptible-minimax.py", EVAL_SIMPLE_MATH),
    ("07z-interruptible-sarvam.py", EVAL_SIMPLE_MATH),
    ("07ae-interruptible-hume.py", EVAL_SIMPLE_MATH),
    # Needs a local XTTS docker instance running.
    # ("07i-interruptible-xtts.py", EVAL_SIMPLE_MATH),
    # Needs a Krisp license.
    # ("07p-interruptible-krisp.py", EVAL_SIMPLE_MATH),
    # Needs GPU resources.
    # ("07u-interruptible-ultravox.py", EVAL_SIMPLE_MATH),
]

TESTS_12 = [
    ("12-describe-image-openai.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("12a-describe-image-anthropic.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("12b-describe-image-aws.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("12c-describe-image-gemini-flash.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("12d-describe-image-moondream.py", EVAL_VISION_IMAGE()),
]

TESTS_14 = [
    ("14-function-calling.py", EVAL_WEATHER),
    ("14a-function-calling-anthropic.py", EVAL_WEATHER),
    ("14e-function-calling-google.py", EVAL_WEATHER),
    ("14f-function-calling-groq.py", EVAL_WEATHER),
    ("14g-function-calling-grok.py", EVAL_WEATHER),
    ("14h-function-calling-azure.py", EVAL_WEATHER),
    ("14i-function-calling-fireworks.py", EVAL_WEATHER),
    ("14j-function-calling-nim.py", EVAL_WEATHER),
    ("14k-function-calling-cerebras.py", EVAL_WEATHER),
    ("14m-function-calling-openrouter.py", EVAL_WEATHER),
    ("14n-function-calling-perplexity.py", EVAL_WEATHER),
    ("14p-function-calling-gemini-vertex-ai.py", EVAL_WEATHER),
    ("14q-function-calling-qwen.py", EVAL_WEATHER),
    ("14r-function-calling-aws.py", EVAL_WEATHER),
    ("14v-function-calling-openai.py", EVAL_WEATHER),
    ("14w-function-calling-mistral.py", EVAL_WEATHER),
    ("14x-function-calling-openpipe.py", EVAL_WEATHER),
    # Video
    ("14d-function-calling-anthropic-video.py", EVAL_VISION_CAMERA),
    ("14d-function-calling-aws-video.py", EVAL_VISION_CAMERA),
    ("14d-function-calling-gemini-flash-video.py", EVAL_VISION_CAMERA),
    ("14d-function-calling-moondream-video.py", EVAL_VISION_CAMERA),
    ("14d-function-calling-openai-video.py", EVAL_VISION_CAMERA),
    # Currently not working.
    # ("14c-function-calling-together.py", EVAL_WEATHER),
    # ("14l-function-calling-deepseek.py", EVAL_WEATHER),
    # ("14o-function-calling-gemini-openai-format.py", EVAL_WEATHER),
]

TESTS_15 = [
    ("15a-switch-languages.py", EVAL_SWITCH_LANGUAGE),
]

TESTS_19 = [
    ("19-openai-realtime.py", EVAL_WEATHER),
    ("19-openai-realtime-beta.py", EVAL_WEATHER),
    # OpenAI Realtime not released on Azure yet
    # ("19a-azure-realtime.py", EVAL_WEATHER),
    ("19a-azure-realtime-beta.py", EVAL_WEATHER),
    ("19b-openai-realtime-text.py", EVAL_WEATHER),
    ("19b-openai-realtime-beta-text.py", EVAL_WEATHER),
]

TESTS_21 = [
    ("21a-tavus-video-service.py", EVAL_SIMPLE_MATH),
]

TESTS_26 = [
    ("26-gemini-live.py", EVAL_SIMPLE_MATH),
    ("26a-gemini-live-transcription.py", EVAL_SIMPLE_MATH),
    ("26b-gemini-live-function-calling.py", EVAL_WEATHER),
    ("26c-gemini-live-video.py", EVAL_VISION_CAMERA),
    ("26e-gemini-live-google-search.py", EVAL_ONLINE_SEARCH),
    ("26h-gemini-live-vertex-function-calling.py", EVAL_WEATHER),
    # Currently not working.
    # ("26d-gemini-live-text.py", EVAL_SIMPLE_MATH),
]

TESTS_27 = [
    ("27-simli-layer.py", EVAL_SIMPLE_MATH),
]

TESTS_40 = [
    ("40-aws-nova-sonic.py", EVAL_SIMPLE_MATH),
]

TESTS_43 = [
    ("43a-heygen-video-service.py", EVAL_SIMPLE_MATH),
]

TESTS_44 = [
    ("44-voicemail-detection.py", EVAL_VOICEMAIL),
    ("44-voicemail-detection.py", EVAL_CONVERSATION),
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

    # Parse test config: (test, prompt, eval, user_speaks_first)
    for test_config in TESTS:
        test, eval_config = test_config

        await runner.run_eval(test, eval_config)

    runner.print_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Eval Runner")
    parser.add_argument("--audio", "-a", action="store_true", help="Record audio for each test")
    parser.add_argument("--name", "-n", help="Name for the current runner (e.g. 'v.0.0.68')")
    parser.add_argument("--pattern", "-p", help="Only run tests that match the pattern")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    asyncio.run(main(args))
