"""Test helper for running imports in a controlled environment.

This script is designed to be called by other testing scripts (e.g.,
`test_all_examples.py`) to isolate and test the importability of Python
modules.

Loguru Conflict Resolution:
The `pipecat-ai` library and its examples use the `loguru` library for logging.
When multiple `pipecat` modules are imported sequentially in the same process,
`loguru`'s default logger configuration can cause conflicts. Some example
modules unconditionally call `logger.remove(0)` to customize the logging setup.
If a previous test step has already removed the default handler, this causes a
`ValueError: There is no existing handler with id 0`.

To prevent this, this script uses `unittest.mock.patch` to temporarily disable
`loguru.logger.remove` during the import of each module. This allows the
modules to be imported without crashing, even if they try to manipulate the
logger in a conflicting way. This is a "hack" to work around the fact that
we cannot modify the example files themselves.

Argument Injection for `custom_track_sender.py`:
The `examples/daily-custom-tracks/custom_track_sender.py` script requires
command-line arguments (`--url` and `--input`) to run. To allow this script to
be imported for testing without crashing, this test helper performs the
following steps specifically for that file:
1.  It detects when `custom_track_sender.py` is being tested.
2.  It generates a small, silent MP3 file using the `pydub` library to serve as
    a valid input file.
3.  It dynamically prepends the required command-line arguments (a dummy URL and
    the path to the silent MP3) to `sys.argv` before importing the module.
4.  After the import, it cleans up by deleting the dummy MP3 file and restoring
    `sys.argv` to its original state.

This ensures the import test can pass without modifying the example's code.

Directory Mocking for `storytelling-chatbot`:
The `examples/storytelling-chatbot/server/bot_runner.py` script expects a
`client/out` directory to exist at startup, as it serves static files from
there. To handle this without requiring a full client-side build, the script:
1.  Detects when `bot_runner.py` is being tested.
2.  Creates the `client/out` directory and a dummy `index.html` file inside it.
3.  Changes the current working directory to the server's directory to ensure
    relative paths are resolved correctly.
4.  After the import, it cleans up by removing the created directory and
    restoring the original working directory.

Monkey-Patching for `07u-interruptible-ultravox.py`:
The `07u-interruptible-ultravox.py` example initializes the `UltravoxModel`,
which in turn initializes the `vLLM` engine. In a CPU-only test environment,
`vLLM` fails to infer the device type and raises an error. To prevent this,
this script patches the `_initialize_engine` method of the `UltravoxModel`
to be a no-op, preventing the engine from being initialized during the import
test.
"""

import importlib.util
import os
import shutil
import sys
from contextlib import ExitStack
from unittest.mock import patch

# Conditionally import loguru
try:
    from loguru import logger

    loguru_available = True
except ImportError:
    loguru_available = False


def main() -> None:
    """Try to import all modules listed on the command line."""
    if len(sys.argv) < 2:
        print(
            "Usage: python test_imports.py"
            " <module_name1> <file_path1> [<module_name2> <file_path2> ...]"
        )
        sys.exit(1)

    # Set a dummy API key to avoid errors in examples that require it.
    os.environ["GOOGLE_API_KEY"] = "dummy_key"

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    # The arguments are pairs of module_name and file_path.
    for i in range(1, len(sys.argv), 2):
        module_name = sys.argv[i]
        file_path = sys.argv[i + 1]

        original_argv = sys.argv.copy()
        original_cwd = os.getcwd()
        dummy_file_path = None
        chatbot_dir_created = False
        try:
            # Default import path is the one we are given.
            import_path = file_path

            if "custom_track_sender.py" in file_path:
                from pydub import AudioSegment

                # Create a dummy mp3 file for the test
                dummy_file_path = "dummy_input.mp3"
                silent_segment = AudioSegment.silent(duration=1000, frame_rate=16000)
                silent_segment.export(dummy_file_path, format="mp3")
                # Prepend dummy arguments for the script
                sys.argv = [
                    sys.argv[0],
                    "-u",
                    "https://example.com/dummy",
                    "-i",
                    dummy_file_path,
                ] + sys.argv[1:]

            if "storytelling-chatbot" in file_path and "bot_runner.py" in file_path:
                server_dir = os.path.dirname(file_path)
                client_out_dir = os.path.join(server_dir, "client", "out")
                os.makedirs(client_out_dir, exist_ok=True)
                chatbot_dir_created = True
                # Create an empty index.html file
                with open(os.path.join(client_out_dir, "index.html"), "w") as f:
                    f.write("")
                os.chdir(server_dir)
                # When we change the CWD, we just want to import the basename.
                import_path = os.path.basename(file_path)

            # Set up a list of context managers (patchers) to apply.
            patchers = []

            # Patch logger.remove to do nothing, to prevent conflicts between examples.
            if loguru_available:
                patchers.append(
                    patch("loguru.logger.remove", lambda *args, **kwargs: None)
                )

            # Monkey-patch UltravoxModel to prevent it from running for the
            # interruptible ultravox example.
            if "07u-interruptible-ultravox.py" in file_path:
                patchers.append(
                    patch(
                        "pipecat.services.ultravox.stt"
                        ".UltravoxModel._initialize_engine",
                        lambda s: None,
                    )
                )

            # Apply all patches.
            with ExitStack() as stack:
                for p in patchers:
                    stack.enter_context(p)

                sys.path.append(os.path.dirname(file_path))
                spec = importlib.util.spec_from_file_location(module_name, import_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    successes.append(file_path)
                else:
                    failures.append((file_path, "Could not create module spec"))

            if (
                "custom_track_sender.py" in file_path
                and dummy_file_path
                and os.path.exists(dummy_file_path)
            ):
                os.remove(dummy_file_path)
            sys.argv = original_argv

        except Exception as e:
            failures.append((file_path, str(e)))
        finally:
            # Restore CWD if we changed it.
            os.chdir(original_cwd)
            # Clean up dummy files and arguments.
            if dummy_file_path and os.path.exists(dummy_file_path):
                os.remove(dummy_file_path)
            if chatbot_dir_created:
                server_dir = os.path.dirname(file_path)
                client_out_dir = os.path.join(server_dir, "client", "out")
                if os.path.exists(client_out_dir):
                    shutil.rmtree(client_out_dir)
            sys.argv = original_argv

    if successes:
        print("--- SUCCESSFUL IMPORTS ---")
        for path in successes:
            print(path)

    if failures:
        print("--- FAILED IMPORTS ---")
        for path, error in failures:
            print(f"{path}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
