#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pathlib import Path

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
CLEAR = "\033[K"


def print_begin_test(example_file: str):
    print(f"{example_file:<25} RUNNING...{CLEAR}", end="\r", flush=True)


def print_end_test(example_file: str, passed: bool):
    status = f"{GREEN}✅ OK{RESET}" if passed else f"{RED}❌ FAILED{RESET}"
    print(f"{example_file:<25} {status}{CLEAR}")


import importlib.util


def load_module_from_path(path: str | Path):
    path = Path(path).resolve()
    module_name = path.stem

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
