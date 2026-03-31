#
# Copyright (c) 2024–2026, Daily
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

FOUNDATIONAL_DIR = SCRIPT_DIR.parent.parent / "examples"

EVAL_SIMPLE_MATH = EvalConfig(
    prompt="A simple math addition.",
    eval="The user answers the math addition correctly.",
)

EVAL_WEATHER = EvalConfig(
    prompt="What's the weather in San Francisco? Temperature should be in Fahrenheit.",
    eval="The user talks about the weather in San Francisco, including the degrees.",
)

EVAL_WEATHER_AND_RESTAURANT = EvalConfig(
    prompt="What's the weather in San Francisco, and what's a good restaurant there? Temperature should be in Fahrenheit.",
    eval="The user talks about the weather in San Francisco, including the degrees, and provides a restaurant recommendation.",
)

EVAL_ONLINE_SEARCH = EvalConfig(
    prompt="What's the current date in UTC?",
    eval=f"Current date in UTC is {datetime.now(timezone.utc).strftime('%A, %B %d, %Y')}.",
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
    eval="The user provides a reasonable voicemail message.",
    eval_speaks_first=True,
)

EVAL_CONVERSATION = EvalConfig(
    prompt="Hello, this is Mark.",
    eval="The user provides any reasonable conversational response to the greeting.",
    eval_speaks_first=True,
)

EVAL_FLIGHT_STATUS = EvalConfig(
    prompt="Check the status of flight AA100.",
    eval="The user says something about the status of flight AA100, such as whether it's on time or delayed.",
)

EVAL_ORDER = EvalConfig(
    prompt="I'd like to order a chocolate iced doughnut and a regular brewed coffee.",
    eval="The user acknowledges the order of a chocolate iced doughnut and regular brewed coffee.",
    eval_speaks_first=True,
)

EVAL_COMPLETE_TURN = EvalConfig(
    prompt="I would go to Japan because I love the culture and want to try authentic ramen.",
    eval="The user provides a relevant response about Japan or travel, showing the conversation continues normally.",
)


TESTS_SPEECH = [
    ("services/speech/cartesia.py", EVAL_SIMPLE_MATH),
    ("services/speech/cartesia-http.py", EVAL_SIMPLE_MATH),
    ("services/speech/speechmatics.py", EVAL_SIMPLE_MATH),
    ("services/speech/speechmatics-vad.py", EVAL_SIMPLE_MATH),
    ("services/speech/langchain.py", EVAL_SIMPLE_MATH),
    ("services/speech/deepgram.py", EVAL_SIMPLE_MATH),
    ("services/speech/deepgram-flux.py", EVAL_SIMPLE_MATH),
    ("services/speech/deepgram-http.py", EVAL_SIMPLE_MATH),
    ("services/speech/elevenlabs.py", EVAL_SIMPLE_MATH),
    ("services/speech/elevenlabs-http.py", EVAL_SIMPLE_MATH),
    ("services/speech/xai.py", EVAL_SIMPLE_MATH),
    ("services/speech/azure.py", EVAL_SIMPLE_MATH),
    ("services/speech/azure-http.py", EVAL_SIMPLE_MATH),
    ("services/speech/openai.py", EVAL_SIMPLE_MATH),
    ("services/speech/openai-http.py", EVAL_SIMPLE_MATH),
    ("services/speech/gladia.py", EVAL_SIMPLE_MATH),
    ("services/speech/gladia-vad.py", EVAL_SIMPLE_MATH),
    ("services/speech/lmnt.py", EVAL_SIMPLE_MATH),
    ("services/speech/groq.py", EVAL_SIMPLE_MATH),
    ("services/speech/aws.py", EVAL_SIMPLE_MATH),
    ("services/speech/aws-strands.py", EVAL_WEATHER),
    ("services/speech/google-gemini-tts.py", EVAL_SIMPLE_MATH),
    ("services/speech/google.py", EVAL_SIMPLE_MATH),
    ("services/speech/google-http.py", EVAL_SIMPLE_MATH),
    ("services/speech/assemblyai.py", EVAL_SIMPLE_MATH),
    ("services/speech/krisp-viva.py", EVAL_SIMPLE_MATH),
    ("services/speech/rime.py", EVAL_SIMPLE_MATH),
    ("services/speech/rime-http.py", EVAL_SIMPLE_MATH),
    ("services/speech/nvidia.py", EVAL_SIMPLE_MATH),
    ("services/speech/google-audio-in.py", EVAL_SIMPLE_MATH),
    ("services/speech/fish.py", EVAL_SIMPLE_MATH),
    ("services/speech/neuphonic.py", EVAL_SIMPLE_MATH),
    ("services/speech/neuphonic-http.py", EVAL_SIMPLE_MATH),
    ("services/speech/fal.py", EVAL_SIMPLE_MATH),
    ("services/speech/minimax.py", EVAL_SIMPLE_MATH),
    ("services/speech/sarvam.py", EVAL_SIMPLE_MATH),
    ("services/speech/sarvam-http.py", EVAL_SIMPLE_MATH),
    ("services/speech/soniox.py", EVAL_SIMPLE_MATH),
    ("services/speech/inworld.py", EVAL_SIMPLE_MATH),
    ("services/speech/inworld-http.py", EVAL_SIMPLE_MATH),
    ("services/speech/asyncai.py", EVAL_SIMPLE_MATH),
    ("services/speech/asyncai-http.py", EVAL_SIMPLE_MATH),
    ("services/speech/aicoustics.py", EVAL_SIMPLE_MATH),
    ("services/speech/hume.py", EVAL_SIMPLE_MATH),
    ("services/speech/gradium.py", EVAL_SIMPLE_MATH),
    ("services/speech/camb.py", EVAL_SIMPLE_MATH),
    ("services/speech/piper.py", EVAL_SIMPLE_MATH),
    ("services/speech/kokoro.py", EVAL_SIMPLE_MATH),
    ("services/speech/resemble.py", EVAL_SIMPLE_MATH),
    ("services/speech/smallest.py", EVAL_SIMPLE_MATH),
    ("services/speech/openai-responses.py", EVAL_SIMPLE_MATH),
    ("services/speech/openai-responses-http.py", EVAL_SIMPLE_MATH),
    # Needs a local XTTS docker instance running.
    # ("services/speech/xtts.py", EVAL_SIMPLE_MATH),
]

TESTS_VISION = [
    ("vision/openai.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("vision/openai-responses.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("vision/openai-responses-http.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("vision/anthropic.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("vision/aws.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("vision/gemini-flash.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("vision/moondream.py", EVAL_VISION_IMAGE()),
]

# For a few major services, we also test parallel function calling.
# (We don't bother doing this with every single service, as it's expensive and
# most rely on the same OpenAI-compatible implementation.)
TESTS_FUNCTION_CALLING = [
    ("getting-started/07-function-calling.py", EVAL_WEATHER),
    ("getting-started/07-function-calling.py", EVAL_WEATHER_AND_RESTAURANT),
    ("services/function-calling/openai-responses.py", EVAL_WEATHER),
    ("services/function-calling/openai-responses.py", EVAL_WEATHER_AND_RESTAURANT),
    ("services/function-calling/openai-responses-http.py", EVAL_WEATHER),
    ("services/function-calling/openai-responses-http.py", EVAL_WEATHER_AND_RESTAURANT),
    ("services/function-calling/anthropic.py", EVAL_WEATHER),
    ("services/function-calling/anthropic.py", EVAL_WEATHER_AND_RESTAURANT),
    ("services/function-calling/openai.py", EVAL_WEATHER),
    ("services/function-calling/google.py", EVAL_WEATHER),
    ("services/function-calling/google.py", EVAL_WEATHER_AND_RESTAURANT),
    ("services/function-calling/groq.py", EVAL_WEATHER),
    ("services/function-calling/grok.py", EVAL_WEATHER),
    ("services/function-calling/azure.py", EVAL_WEATHER),
    ("services/function-calling/fireworks.py", EVAL_WEATHER),
    ("services/function-calling/nvidia.py", EVAL_WEATHER),
    ("services/function-calling/cerebras.py", EVAL_WEATHER),
    ("services/function-calling/openrouter.py", EVAL_WEATHER),
    ("services/function-calling/perplexity.py", EVAL_WEATHER),
    ("services/function-calling/google-vertex.py", EVAL_WEATHER),
    ("services/function-calling/qwen.py", EVAL_WEATHER),
    ("services/function-calling/aws.py", EVAL_WEATHER),
    ("services/function-calling/sambanova.py", EVAL_WEATHER),
    ("services/function-calling/aws.py", EVAL_WEATHER_AND_RESTAURANT),
    ("services/function-calling/nebius.py", EVAL_WEATHER),
    ("services/function-calling/mistral.py", EVAL_WEATHER),
    ("services/function-calling/sarvam.py", EVAL_WEATHER),
    ("services/function-calling/novita.py", EVAL_WEATHER),
    # Video
    ("services/function-calling/anthropic-video.py", EVAL_VISION_CAMERA),
    ("services/function-calling/aws-video.py", EVAL_VISION_CAMERA),
    ("services/function-calling/google-video.py", EVAL_VISION_CAMERA),
    ("services/function-calling/moondream-video.py", EVAL_VISION_CAMERA),
    ("services/function-calling/openai-video.py", EVAL_VISION_CAMERA),
    ("services/function-calling/openai-responses-video.py", EVAL_VISION_CAMERA),
    ("services/function-calling/openai-responses-video-http.py", EVAL_VISION_CAMERA),
    # Currently not working.
    # ("services/function-calling/together.py", EVAL_WEATHER),
    # ("services/function-calling/deepseek.py", EVAL_WEATHER),
    # ("services/function-calling/gemini-openai-format.py", EVAL_WEATHER),
]

TESTS_FEATURES = [
    ("features/switch-languages.py", EVAL_SWITCH_LANGUAGE),
    ("features/voicemail-detection.py", EVAL_VOICEMAIL),
    ("features/voicemail-detection.py", EVAL_CONVERSATION),
    ("features/concurrent-llm-evaluation.py", EVAL_SIMPLE_MATH),
]

TESTS_REALTIME = [
    ("realtime/openai.py", EVAL_WEATHER),
    ("realtime/openai-beta.py", EVAL_WEATHER),
    # OpenAI Realtime not released on Azure yet
    # ("realtime/azure.py", EVAL_WEATHER),
    ("realtime/azure-beta.py", EVAL_WEATHER),
    ("realtime/openai-text.py", EVAL_WEATHER),
    ("realtime/openai-beta-text.py", EVAL_WEATHER),
    ("realtime/openai-live-video.py", EVAL_VISION_CAMERA),
    ("realtime/gemini-live.py", EVAL_SIMPLE_MATH),
    ("realtime/gemini-live-local-vad.py", EVAL_SIMPLE_MATH),
    ("realtime/gemini-live-function-calling.py", EVAL_WEATHER),
    ("realtime/gemini-live-video.py", EVAL_VISION_CAMERA),
    ("realtime/gemini-live-google-search.py", EVAL_ONLINE_SEARCH),
    ("realtime/gemini-live-vertex-function-calling.py", EVAL_WEATHER),
    # Currently not working.
    # ("realtime/gemini-live-text.py", EVAL_SIMPLE_MATH),
    ("realtime/aws-nova-sonic.py", EVAL_SIMPLE_MATH),
    ("realtime/ultravox.py", EVAL_ORDER),
    ("realtime/grok.py", EVAL_WEATHER),
]

TESTS_VIDEO_AVATAR = [
    ("video-avatar/tavus-video-service.py", EVAL_SIMPLE_MATH),
    ("video-avatar/heygen-video-service.py", EVAL_SIMPLE_MATH),
    ("video-avatar/simli-layer.py", EVAL_SIMPLE_MATH),
    ("video-avatar/lemonslice-transport.py", EVAL_SIMPLE_MATH),
]

TESTS_TURN_MANAGEMENT = [
    ("turn-management/filter-incomplete-turns.py", EVAL_COMPLETE_TURN),
]

TESTS_THINKING_AND_MCP = [
    ("thinking-and-mcp/thinking-anthropic.py", EVAL_SIMPLE_MATH),
    ("thinking-and-mcp/thinking-google.py", EVAL_SIMPLE_MATH),
    ("thinking-and-mcp/thinking-functions-anthropic.py", EVAL_FLIGHT_STATUS),
    ("thinking-and-mcp/thinking-functions-google.py", EVAL_FLIGHT_STATUS),
]

TESTS = [
    *TESTS_SPEECH,
    *TESTS_VISION,
    *TESTS_FUNCTION_CALLING,
    *TESTS_FEATURES,
    *TESTS_REALTIME,
    *TESTS_VIDEO_AVATAR,
    *TESTS_TURN_MANAGEMENT,
    *TESTS_THINKING_AND_MCP,
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
