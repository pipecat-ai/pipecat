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


TESTS_VOICE = [
    ("voice/voice-cartesia.py", EVAL_SIMPLE_MATH),
    ("voice/voice-cartesia-http.py", EVAL_SIMPLE_MATH),
    ("voice/voice-speechmatics.py", EVAL_SIMPLE_MATH),
    ("voice/voice-speechmatics-vad.py", EVAL_SIMPLE_MATH),
    ("voice/voice-langchain.py", EVAL_SIMPLE_MATH),
    ("voice/voice-deepgram.py", EVAL_SIMPLE_MATH),
    ("voice/voice-deepgram-flux.py", EVAL_SIMPLE_MATH),
    ("voice/voice-deepgram-http.py", EVAL_SIMPLE_MATH),
    ("voice/voice-elevenlabs.py", EVAL_SIMPLE_MATH),
    ("voice/voice-elevenlabs-http.py", EVAL_SIMPLE_MATH),
    ("voice/voice-xai.py", EVAL_SIMPLE_MATH),
    ("voice/voice-azure.py", EVAL_SIMPLE_MATH),
    ("voice/voice-azure-http.py", EVAL_SIMPLE_MATH),
    ("voice/voice-openai.py", EVAL_SIMPLE_MATH),
    ("voice/voice-openai-http.py", EVAL_SIMPLE_MATH),
    ("voice/voice-gladia.py", EVAL_SIMPLE_MATH),
    ("voice/voice-gladia-vad.py", EVAL_SIMPLE_MATH),
    ("voice/voice-lmnt.py", EVAL_SIMPLE_MATH),
    ("voice/voice-groq.py", EVAL_SIMPLE_MATH),
    ("voice/voice-aws.py", EVAL_SIMPLE_MATH),
    ("voice/voice-aws-strands.py", EVAL_WEATHER),
    ("voice/voice-google-gemini-tts.py", EVAL_SIMPLE_MATH),
    ("voice/voice-google.py", EVAL_SIMPLE_MATH),
    ("voice/voice-google-http.py", EVAL_SIMPLE_MATH),
    ("voice/voice-assemblyai.py", EVAL_SIMPLE_MATH),
    ("voice/voice-krisp-viva.py", EVAL_SIMPLE_MATH),
    ("voice/voice-rime.py", EVAL_SIMPLE_MATH),
    ("voice/voice-rime-http.py", EVAL_SIMPLE_MATH),
    ("voice/voice-nvidia.py", EVAL_SIMPLE_MATH),
    ("voice/voice-google-audio-in.py", EVAL_SIMPLE_MATH),
    ("voice/voice-fish.py", EVAL_SIMPLE_MATH),
    ("voice/voice-neuphonic.py", EVAL_SIMPLE_MATH),
    ("voice/voice-neuphonic-http.py", EVAL_SIMPLE_MATH),
    ("voice/voice-fal.py", EVAL_SIMPLE_MATH),
    ("voice/voice-minimax.py", EVAL_SIMPLE_MATH),
    ("voice/voice-sarvam.py", EVAL_SIMPLE_MATH),
    ("voice/voice-sarvam-http.py", EVAL_SIMPLE_MATH),
    ("voice/voice-soniox.py", EVAL_SIMPLE_MATH),
    ("voice/voice-inworld.py", EVAL_SIMPLE_MATH),
    ("voice/voice-inworld-http.py", EVAL_SIMPLE_MATH),
    ("voice/voice-asyncai.py", EVAL_SIMPLE_MATH),
    ("voice/voice-asyncai-http.py", EVAL_SIMPLE_MATH),
    ("voice/voice-aicoustics.py", EVAL_SIMPLE_MATH),
    ("voice/voice-hume.py", EVAL_SIMPLE_MATH),
    ("voice/voice-gradium.py", EVAL_SIMPLE_MATH),
    ("voice/voice-camb.py", EVAL_SIMPLE_MATH),
    ("voice/voice-piper.py", EVAL_SIMPLE_MATH),
    ("voice/voice-kokoro.py", EVAL_SIMPLE_MATH),
    ("voice/voice-resemble.py", EVAL_SIMPLE_MATH),
    ("voice/voice-smallest.py", EVAL_SIMPLE_MATH),
    ("voice/voice-mistral.py", EVAL_SIMPLE_MATH),
    ("voice/voice-openai-responses.py", EVAL_SIMPLE_MATH),
    ("voice/voice-openai-responses-http.py", EVAL_SIMPLE_MATH),
    # Needs a local XTTS docker instance running.
    # ("voice/voice-xtts.py", EVAL_SIMPLE_MATH),
]

TESTS_VISION = [
    ("vision/vision-openai.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("vision/vision-openai-responses.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("vision/vision-openai-responses-http.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("vision/vision-anthropic.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("vision/vision-aws.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("vision/vision-gemini-flash.py", EVAL_VISION_IMAGE(eval_speaks_first=True)),
    ("vision/vision-moondream.py", EVAL_VISION_IMAGE()),
]

# For a few major services, we also test parallel function calling.
# (We don't bother doing this with every single service, as it's expensive and
# most rely on the same OpenAI-compatible implementation.)
TESTS_FUNCTION_CALLING = [
    ("getting-started/07-function-calling.py", EVAL_WEATHER),
    ("getting-started/07-function-calling.py", EVAL_WEATHER_AND_RESTAURANT),
    ("function-calling/function-calling-openai-responses.py", EVAL_WEATHER),
    ("function-calling/function-calling-openai-responses.py", EVAL_WEATHER_AND_RESTAURANT),
    ("function-calling/function-calling-openai-responses-http.py", EVAL_WEATHER),
    ("function-calling/function-calling-openai-responses-http.py", EVAL_WEATHER_AND_RESTAURANT),
    ("function-calling/function-calling-anthropic.py", EVAL_WEATHER),
    ("function-calling/function-calling-anthropic.py", EVAL_WEATHER_AND_RESTAURANT),
    ("function-calling/function-calling-openai.py", EVAL_WEATHER),
    ("function-calling/function-calling-google.py", EVAL_WEATHER),
    ("function-calling/function-calling-google.py", EVAL_WEATHER_AND_RESTAURANT),
    ("function-calling/function-calling-groq.py", EVAL_WEATHER),
    ("function-calling/function-calling-grok.py", EVAL_WEATHER),
    ("function-calling/function-calling-azure.py", EVAL_WEATHER),
    ("function-calling/function-calling-fireworks.py", EVAL_WEATHER),
    ("function-calling/function-calling-nvidia.py", EVAL_WEATHER),
    ("function-calling/function-calling-cerebras.py", EVAL_WEATHER),
    ("function-calling/function-calling-openrouter.py", EVAL_WEATHER),
    ("function-calling/function-calling-perplexity.py", EVAL_WEATHER),
    ("function-calling/function-calling-google-vertex.py", EVAL_WEATHER),
    ("function-calling/function-calling-qwen.py", EVAL_WEATHER),
    ("function-calling/function-calling-aws.py", EVAL_WEATHER),
    ("function-calling/function-calling-sambanova.py", EVAL_WEATHER),
    ("function-calling/function-calling-aws.py", EVAL_WEATHER_AND_RESTAURANT),
    ("function-calling/function-calling-nebius.py", EVAL_WEATHER),
    ("function-calling/function-calling-mistral.py", EVAL_WEATHER),
    ("function-calling/function-calling-sarvam.py", EVAL_WEATHER),
    ("function-calling/function-calling-novita.py", EVAL_WEATHER),
    ("function-calling/function-calling-deepseek.py", EVAL_WEATHER),
    # Video
    ("function-calling/function-calling-anthropic-video.py", EVAL_VISION_CAMERA),
    ("function-calling/function-calling-aws-video.py", EVAL_VISION_CAMERA),
    ("function-calling/function-calling-google-video.py", EVAL_VISION_CAMERA),
    ("function-calling/function-calling-moondream-video.py", EVAL_VISION_CAMERA),
    ("function-calling/function-calling-openai-video.py", EVAL_VISION_CAMERA),
    ("function-calling/function-calling-openai-responses-video.py", EVAL_VISION_CAMERA),
    ("function-calling/function-calling-openai-responses-video-http.py", EVAL_VISION_CAMERA),
    # Currently not working.
    # ("function-calling/function-calling-together.py", EVAL_WEATHER),
]

TESTS_FEATURES = [
    ("features/features-switch-languages.py", EVAL_SWITCH_LANGUAGE),
    ("features/features-voicemail-detection.py", EVAL_VOICEMAIL),
    ("features/features-voicemail-detection.py", EVAL_CONVERSATION),
    ("features/features-concurrent-llm-evaluation.py", EVAL_SIMPLE_MATH),
]

TESTS_REALTIME = [
    ("realtime/realtime-openai.py", EVAL_WEATHER),
    # OpenAI Realtime not released on Azure yet
    # ("realtime/realtime-azure.py", EVAL_WEATHER),
    ("realtime/realtime-openai-text.py", EVAL_WEATHER),
    ("realtime/realtime-openai-live-video.py", EVAL_VISION_CAMERA),
    ("realtime/realtime-gemini-live.py", EVAL_SIMPLE_MATH),
    ("realtime/realtime-gemini-live-local-vad.py", EVAL_SIMPLE_MATH),
    ("realtime/realtime-gemini-live-function-calling.py", EVAL_WEATHER),
    ("realtime/realtime-gemini-live-video.py", EVAL_VISION_CAMERA),
    ("realtime/realtime-gemini-live-google-search.py", EVAL_ONLINE_SEARCH),
    ("realtime/realtime-gemini-live-vertex-function-calling.py", EVAL_WEATHER),
    ("realtime/realtime-aws-nova-sonic.py", EVAL_SIMPLE_MATH),
    ("realtime/realtime-ultravox.py", EVAL_ORDER),
    ("realtime/realtime-grok.py", EVAL_WEATHER),
]

TESTS_VIDEO_AVATAR = [
    ("video-avatar/video-avatar-tavus-video-service.py", EVAL_SIMPLE_MATH),
    ("video-avatar/video-avatar-heygen-video-service.py", EVAL_SIMPLE_MATH),
    ("video-avatar/video-avatar-simli-video-service.py", EVAL_SIMPLE_MATH),
    ("video-avatar/video-avatar-lemonslice-transport.py", EVAL_SIMPLE_MATH),
]

TESTS_TURN_MANAGEMENT = [
    ("turn-management/turn-management-filter-incomplete-turns.py", EVAL_COMPLETE_TURN),
]

TESTS_THINKING = [
    ("thinking/thinking-anthropic.py", EVAL_SIMPLE_MATH),
    ("thinking/thinking-google.py", EVAL_SIMPLE_MATH),
    ("thinking/thinking-functions-anthropic.py", EVAL_FLIGHT_STATUS),
    ("thinking/thinking-functions-google.py", EVAL_FLIGHT_STATUS),
]

TESTS = [
    *TESTS_VOICE,
    *TESTS_VISION,
    *TESTS_FUNCTION_CALLING,
    *TESTS_FEATURES,
    *TESTS_REALTIME,
    *TESTS_VIDEO_AVATAR,
    *TESTS_TURN_MANAGEMENT,
    *TESTS_THINKING,
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

    concurrency = args.concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async def run_with_semaphore(test: str, eval_config: EvalConfig):
        async with semaphore:
            await runner.run_eval(test, eval_config)

    tasks = []
    for test, eval_config in TESTS:
        tasks.append(asyncio.create_task(run_with_semaphore(test, eval_config)))

    await asyncio.gather(*tasks)

    runner.print_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Eval Runner")
    parser.add_argument("--audio", "-a", action="store_true", help="Record audio for each test")
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=3,
        help="Max number of evals to run concurrently (default: 3)",
    )
    parser.add_argument("--name", "-n", help="Name for the current runner (e.g. 'v.0.0.68')")
    parser.add_argument("--pattern", "-p", help="Only run tests that match the pattern")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    asyncio.run(main(args))
