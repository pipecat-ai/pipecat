#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import importlib
import os


def get_bot_file(arg_bot: str | None) -> str:
    bot_implementation = arg_bot or os.getenv("BOT_IMPLEMENTATION", "openai").lower().strip()
    if not bot_implementation:
        bot_implementation = "openai"
    if bot_implementation not in ["openai", "gemini", "vllm"]:
        raise ValueError(
            f"Invalid BOT_IMPLEMENTATION: {bot_implementation}. Must be 'openai' or 'gemini'"
        )
    return f"bot_{bot_implementation}"


def get_runner(bot_file: str):
    """Dynamically import the run_bot function based on the bot name.

    Args:
        bot_name (str): The name of the bot implementation (e.g., 'openai', 'gemini').

    Returns:
        function: The run_bot function from the specified bot module.

    Raises:
        ImportError: If the specified bot module or run_bot function is not found.
    """
    try:
        # Dynamically construct the module name
        module_name = f"{bot_file}"
        # Import the module
        module = importlib.import_module(module_name)
        # Get the run_bot function from the module
        return getattr(module, "run_bot")
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import run_bot from {module_name}: {e}")


def main():
    """Parse the args to launch the appropriate bot using the given room/token."""
    parser = argparse.ArgumentParser(description="Daily AI SDK Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="URL of the Daily room to join"
    )
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        required=False,
        help="Daily room token",
    )
    parser.add_argument(
        "-b",
        "--bot",
        type=str,
        required=False,
        help="Bot runner to use (e.g., openai, gemini)",
    )

    args, unknown = parser.parse_known_args()

    url = args.url or os.getenv("DAILY_SAMPLE_ROOM_URL")
    token = args.token or os.getenv("DAILY_SAMPLE_ROOM_TOKEN")
    bot_file = get_bot_file(args.bot)

    if not url:
        raise Exception(
            "No Daily room specified. use the -u/--url option from the command line, or set DAILY_SAMPLE_ROOM_URL in your environment to specify a Daily room URL."
        )

    run_bot = get_runner(bot_file)
    asyncio.run(run_bot(url, token))


if __name__ == "__main__":
    main()
