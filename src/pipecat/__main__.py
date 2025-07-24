#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from loguru import logger


def load_bot_module(file_path: str, function_name: str = "run_example"):
    """Load a bot module from a Python file and return the specified function.

    Args:
        file_path: Path to the Python file containing the bot
        function_name: Name of the function to load (default: run_example)

    Returns:
        The callable function from the module

    Raises:
        SystemExit: If the file doesn't exist, isn't a Python file, or the function isn't found
    """
    logger.info(f"Loading bot module from: {file_path}")
    logger.info(f"Looking for function: {function_name}")

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        print(f"Error: File '{file_path}' not found", file=sys.stderr)
        sys.exit(1)

    if not file_path_obj.suffix == ".py":
        print(f"Error: File '{file_path}' is not a Python file", file=sys.stderr)
        sys.exit(1)

    # Import the module
    try:
        logger.info(f"Importing module from: {file_path}")
        spec = importlib.util.spec_from_file_location("bot_module", file_path_obj)
        if spec is None or spec.loader is None:
            print(f"Error: Could not load module from '{file_path}'", file=sys.stderr)
            sys.exit(1)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info(f"Successfully imported module: {module.__name__}")
    except Exception as e:
        print(f"Error importing module from '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Find the function to run
    if not hasattr(module, function_name):
        print(f"Error: Function '{function_name}' not found in '{file_path}'", file=sys.stderr)
        print(
            f"Available functions: {[name for name in dir(module) if not name.startswith('_')]}", file=sys.stderr)
        sys.exit(1)

    run_example = getattr(module, function_name)
    if not callable(run_example):
        print(f"Error: '{function_name}' is not a callable function", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Successfully loaded function: {function_name}")
    return run_example


def main():
    """Main entry point for the pipecat command line tool.

    This function is called by the entry point script and handles argument parsing
    and module loading before calling the actual main execution logic.
    """
    # Set up argument parser for our specific arguments
    parser = argparse.ArgumentParser(description="Run a Pipecat bot from a Python file")
    parser.add_argument("file", help="Python file containing the bot to run")
    parser.add_argument("--function", "-f", default="run_example",
                        help="Function name to run (default: run_example)")

    # Parse our arguments first
    args, remaining_args = parser.parse_known_args()

    # Load the bot module and get the function
    run_example = load_bot_module(args.file, args.function)

    # Set sys.argv to the remaining arguments for the run_main function
    sys.argv = [sys.argv[0]] + remaining_args

    # Import run_main only when we need it
    from pipecat.examples.run import main as run_main

    # Call the main function from pipecat.examples.run
    run_main(run_example)


if __name__ == "__main__":
    main()
