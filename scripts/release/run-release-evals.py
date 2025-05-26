#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import sys

from dotenv import load_dotenv
from eval import run_eval
from loguru import logger

load_dotenv(override=True)

# Math
PROMPT_SIMPLE_MATH = "A simple math addition."

# Weather
PROMPT_WEATHER = "What's the weather in San Francisco?"
EVAL_WEATHER = "Something specific about the current weather in San Francisco."


async def run_07_evals():
    await run_eval("07-interruptible.py", PROMPT_SIMPLE_MATH)
    await run_eval("07-interruptible-cartesia-http.py", PROMPT_SIMPLE_MATH)
    await run_eval("07b-interruptible-langchain.py", PROMPT_SIMPLE_MATH)
    await run_eval("07c-interruptible-deepgram.py", PROMPT_SIMPLE_MATH)
    await run_eval("07d-interruptible-elevenlabs.py", PROMPT_SIMPLE_MATH)
    await run_eval("07d-interruptible-elevenlabs-http.py", PROMPT_SIMPLE_MATH)
    await run_eval("07e-interruptible-playht.py", PROMPT_SIMPLE_MATH)


async def run_14_evals():
    await run_eval("14-function-calling.py", PROMPT_WEATHER, EVAL_WEATHER)
    await run_eval("14a-function-calling-anthropic.py", PROMPT_WEATHER, EVAL_WEATHER)


async def main():
    await run_07_evals()
    await run_14_evals()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Eval Runner")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    # Log level
    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="TRACE" if args.verbose >= 2 else "DEBUG")

    asyncio.run(main())
