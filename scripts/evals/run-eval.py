#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from eval import EvalRunner
from loguru import logger
from utils import check_env_variables

load_dotenv(override=True)


async def main(args: argparse.Namespace):
    if not check_env_variables():
        return

    # Log level
    logger.remove(0)
    log_level = "TRACE" if args.verbose >= 2 else "DEBUG"
    if args.verbose:
        logger.add(sys.stderr, level=log_level)

    script_path = Path(os.path.dirname(os.path.abspath(args.script)))
    script_file = os.path.basename(args.script)

    runner = EvalRunner(examples_dir=script_path, record_audio=args.audio, log_level=log_level)

    await runner.run_eval(script_file, args.prompt, args.eval)

    runner.print_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Eval Runner")
    parser.add_argument("--audio", "-a", action="store_true", help="Record audio for each test")
    parser.add_argument("--prompt", "-p", required=True, help="Prompt for this eval")
    parser.add_argument("--eval", "-e", required=False, help="Eval verification")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("script", help="Script to run")
    args = parser.parse_args()

    asyncio.run(main(args))
