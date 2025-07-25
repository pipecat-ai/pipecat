#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import importlib.util
import os
import sys


def find_and_import_bot():
    """Find and import the bot function from the current working directory."""
    cwd = os.getcwd()

    # Add current working directory to Python path if not already there
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    # Try to find bot.py in current directory
    bot_file = os.path.join(cwd, "bot.py")
    if os.path.exists(bot_file):
        spec = importlib.util.spec_from_file_location("bot", bot_file)
        bot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bot_module)
        return bot_module

    # Try to import bot module directly
    try:
        import bot

        return bot
    except ImportError:
        pass

    # Look for any .py file in current directory that has a bot function
    for filename in os.listdir(cwd):
        if filename.endswith(".py") and filename != "runner.py":
            try:
                module_name = filename[:-3]  # Remove .py extension
                spec = importlib.util.spec_from_file_location(
                    module_name, os.path.join(cwd, filename)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "bot"):
                    return module
            except Exception:
                continue

    raise ImportError(
        "Could not find 'bot' function. Make sure your bot file has a 'bot' function."
    )


def main():
    """Parse args and launch the bot with specified transport."""
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument("-u", "--url", type=str, required=False, help="Daily room URL")
    parser.add_argument("-t", "--token", type=str, required=False, help="Daily room token")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["daily", "livekit", "webrtc"],
        default="daily",
        help="Transport type",
    )
    parser.add_argument("--room", type=str, required=False, help="LiveKit room name")

    args, unknown = parser.parse_known_args()

    # Find and import the bot function
    try:
        bot_module = find_and_import_bot()
        bot_function = bot_module.bot
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.transport == "daily":
        url = args.url or os.getenv("DAILY_SAMPLE_ROOM_URL")
        token = args.token or os.getenv("DAILY_SAMPLE_ROOM_TOKEN")

        if not url or not token:
            raise Exception("Daily room URL and token are required.")

        # Create Daily session arguments
        try:
            from pipecatcloud.agent import DailySessionArguments

            session_args = DailySessionArguments(
                room_url=url,
                token=token,
                body={},
                session_id=None,
            )
        except ImportError:
            # Fallback for local development
            class LocalDailySessionArgs:
                def __init__(self, room_url, token, body=None):
                    self.room_url = room_url
                    self.token = token
                    self.body = body or {}

            session_args = LocalDailySessionArgs(url, token)

    elif args.transport == "livekit":
        url = args.url or os.getenv("LIVEKIT_URL")
        token = args.token or os.getenv("LIVEKIT_TOKEN")
        room_name = args.room or os.getenv("LIVEKIT_ROOM_NAME")

        if not url or not token or not room_name:
            raise Exception("LiveKit URL, token, and room name are required.")

        class LiveKitSessionArgs:
            def __init__(self, url, token, room_name):
                self.url = url
                self.token = token
                self.room_name = room_name
                self.body = {}

        session_args = LiveKitSessionArgs(url, token, room_name)

    elif args.transport == "webrtc":
        # For WebRTC subprocess mode (not typically used)
        class WebRTCSessionArgs:
            def __init__(self):
                self.transport_type = "webrtc"
                self.body = {}

        session_args = WebRTCSessionArgs()

    else:
        raise Exception(f"Unsupported transport: {args.transport}")

    asyncio.run(bot_function(session_args))


if __name__ == "__main__":
    main()
