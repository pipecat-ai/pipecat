#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
CLEAR = "\033[K"


@dataclass
class EvalResult:
    name: str
    result: bool
    time: float


def check_env_variables() -> bool:
    required_envs = [
        "CARTESIA_API_KEY",
        "DEEPGRAM_API_KEY",
        "OPENAI_API_KEY",
        "DAILY_SAMPLE_ROOM_URL",
    ]
    for env in required_envs:
        if not os.getenv(env):
            print(f"\nERROR: Environment variable {env} is not defined.\n")
            print(f"Required environment variables: {required_envs}")
            return False
    return True


def print_begin_test(example_file: str):
    print(f"{example_file:<55} RUNNING...{CLEAR}", end="\r", flush=True)


def print_end_test(example_file: str, passed: bool, time: float):
    status = f"{GREEN}✅ OK{RESET}" if passed else f"{RED}❌ FAILED{RESET}"
    print(f"{example_file:<55} {status} ({time:.2f}s){CLEAR}")


def print_test_results(tests: Sequence[EvalResult], total_success: int, location: str):
    total_count = len(tests)

    bar = "=" * 80

    print()
    print(f"{GREEN}{bar}{RESET}")
    print(f"TOTAL NUMBER OF TESTS: {total_count}")
    print()

    total_time = 0.0
    total_count = len(tests)
    for eval in tests:
        total_time += eval.time
        print_end_test(eval.name, eval.result, eval.time)

    total_fail = total_count - total_success

    print()
    print(
        f"{GREEN}SUCCESS{RESET}: {total_success} | {RED}FAIL{RESET}: {total_fail} | TOTAL TIME: {total_time:.2f}s"
    )
    print(f"{GREEN}{bar}{RESET}")
    print()
    print(f"Tests output: {location}")


def load_module_from_path(path: str | Path):
    path = Path(path).resolve()
    module_name = path.stem

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
