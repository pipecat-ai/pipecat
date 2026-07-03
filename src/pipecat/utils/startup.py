#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared loader for ``PIPECAT_SETUP_FILES`` hooks.

Each file listed in the ``PIPECAT_SETUP_FILES`` environment variable (colon
separated) may define one or both of the following async functions:

- ``setup_worker_runner(runner)`` — invoked once per :class:`WorkerRunner`
  before its spawned workers start.
- ``setup_pipeline_worker(worker)`` — invoked once per :class:`PipelineWorker` while
  the worker sets up its pipeline. The legacy name ``setup_pipeline_task`` is
  still recognized but emits a ``DeprecationWarning``; rename it to
  ``setup_pipeline_worker``.

Setup files are imported at most once per process; module-level state (for
example, a shared debugger instance referenced by both hooks) is preserved
across hook invocations.
"""

import importlib.util
import os
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

from loguru import logger

_module_cache: dict[str, ModuleType] = {}


def _setup_file_paths() -> list[Path]:
    return [Path(f).resolve() for f in os.environ.get("PIPECAT_SETUP_FILES", "").split(":") if f]


def _load_module(path: Path) -> ModuleType | None:
    cache_key = str(path)
    cached = _module_cache.get(cache_key)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        logger.error(f"unable to load setup file {path}")
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.error(f"error loading setup file {path}: {e}")
        return None

    _module_cache[cache_key] = module
    return module


async def run_setup_hook(
    *,
    target: Any,
    function_name: str,
    deprecated_function_name: str | None = None,
) -> None:
    """Run ``function_name(target)`` from every ``PIPECAT_SETUP_FILES`` module.

    Args:
        target: Object passed to the hook function.
        function_name: Name of the async function to call in each setup file.
        deprecated_function_name: Optional legacy name to fall back to when
            ``function_name`` is not defined. When invoked, a
            ``DeprecationWarning`` is emitted telling the user to rename.
    """
    for path in _setup_file_paths():
        module = _load_module(path)
        if module is None:
            continue
        try:
            if hasattr(module, function_name):
                logger.debug(f"{target} running {function_name} from {path}")
                await getattr(module, function_name)(target)
            elif deprecated_function_name and hasattr(module, deprecated_function_name):
                warnings.warn(
                    f"setup file {path} defines '{deprecated_function_name}'; "
                    f"rename it to '{function_name}'. The old name will be removed "
                    f"in 2.0.0.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                logger.debug(f"{target} running {deprecated_function_name} from {path}")
                await getattr(module, deprecated_function_name)(target)
            else:
                logger.warning(f"{target} setup file {path} has no {function_name} function")
        except Exception as e:
            logger.error(f"{target} error running {function_name} from {path}: {e}")
