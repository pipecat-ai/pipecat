#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("pipecat-server")


def list_available_bots():
    """List all available bot files in the bots directory."""
    bots_dir = Path(__file__).parent / "bots"
    bot_files = list(bots_dir.glob("*.py"))

    # Skip __init__.py and other special files
    bot_files = [f for f in bot_files if not f.name.startswith("__")]

    return sorted(bot_files)


def main():
    """Run the Pipecat foundational example web app."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Pipecat Foundational Examples")

    # Bot file argument as positional argument
    parser.add_argument(
        "bot_file",
        nargs="?",
        help="Path to bot Python file to run",
    )

    # Server configuration
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to run the server on (default: 8000)",
    )

    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind the server to (default: 0.0.0.0)",
    )

    # List available bots
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available bots and exit",
    )

    # Parse arguments
    args = parser.parse_args()

    # If list flag is provided, show available bots and exit
    if args.list:
        bot_files = list_available_bots()

        if not bot_files:
            print("No bot files found in the 'bots' directory.")
            sys.exit(1)

        print("\nAvailable Pipecat Bots:")
        print("=========================")

        for bot_file in bot_files:
            print(f"  {bot_file.relative_to(Path(__file__).parent)}")

        print("\nRun with: python run.py <bot-file>")
        sys.exit(0)

    # If no bot file is specified and not listing, show error
    if not args.bot_file:
        parser.error("You must specify a bot file to run. Use --list to see available options.")

    # Get the absolute path to the bot file
    bot_path = Path(args.bot_file).absolute()
    if not bot_path.exists():
        parser.error(f"Bot file not found: {bot_path}")

    # Set up the environment variables for the server
    os.environ["PORT"] = str(args.port)
    os.environ["HOST"] = args.host

    # Store the bot file path in the environment for app.py to access
    os.environ["BOT_FILE_PATH"] = str(bot_path)

    # Print startup information
    logger.info("-" * 43)
    logger.info(f"Starting {bot_path.name}")
    logger.info(f"Open your browser to: http://localhost:{args.port}")
    logger.info("-" * 43)

    # Import the app and run it
    from app import app

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
