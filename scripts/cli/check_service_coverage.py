#!/usr/bin/env python3
"""Check that the CLI registry accounts for every service class.

Imports every module under ``pipecat.services`` and finds the concrete STT, LLM and
TTS service classes (realtime services are LLM services). Each one must appear in a
service list in ``service_metadata.py`` or in ``EXCLUDED_SERVICES``. The check also
fails when the registry offers a deprecated class, or when ``EXCLUDED_SERVICES``
names a class that isn't a concrete service.

These classes aren't concrete services, so they're skipped:

- abstract classes, classes defined in a top-level ``pipecat.services`` module, and
  classes named as bases (``_*``, ``Base*``, ``*Base``)
- deprecated classes
- the HTTP version of a service the provider also streams

Importing every service module needs every extra installed, so this runs in CI
rather than as a pre-commit hook. A module that fails to import fails the check.

Usage:
    uv run scripts/cli/check_service_coverage.py
"""

import importlib
import inspect
import pkgutil
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pipecat.services
from pipecat.cli.registry.service_metadata import EXCLUDED_SERVICES, ServiceRegistry
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

_BASE_NAME = re.compile(r"^_|Base[A-Z]|Base$")
_HTTP_NAME = re.compile(r"Http|HTTP")


def import_service_modules() -> tuple[list, list[str]]:
    """Import every module under ``pipecat.services``.

    Returns:
        The imported modules, and a description of each module that failed.
    """
    modules = []
    failures = []

    def on_package_error(name: str) -> None:
        failures.append(f"{name}: {sys.exc_info()[1]}")

    # Deprecated modules warn on import; recording the warnings keeps them quiet.
    with warnings.catch_warnings(record=True):
        for info in pkgutil.walk_packages(
            pipecat.services.__path__, "pipecat.services.", onerror=on_package_error
        ):
            try:
                modules.append(importlib.import_module(info.name))
            except Exception as e:
                failures.append(f"{info.name}: {type(e).__name__}: {e}")
    return modules, failures


def service_type(cls: type) -> type | None:
    """Return the service base class a class extends, or None if it isn't a service."""
    for base in (STTService, TTSService, LLMService):
        if issubclass(cls, base):
            return base
    return None


def is_base_class(cls: type) -> bool:
    """Whether a class is a base for other services rather than a service itself."""
    return bool(
        inspect.isabstract(cls)
        or cls.__module__.count(".") == 2  # pipecat.services.<module>: framework bases
        or _BASE_NAME.search(cls.__name__)
    )


def is_deprecated(cls: type) -> bool:
    """Whether a class is marked with the ``@deprecated`` decorator."""
    return getattr(cls, "__deprecated__", None) is not None


def is_http_variant(cls: type, services: list[type]) -> bool:
    """Whether a class is the HTTP version of a service its provider also streams."""
    if not _HTTP_NAME.search(cls.__name__):
        return False
    provider = cls.__module__.split(".")[2]
    return any(
        other.__module__.split(".")[2] == provider
        and service_type(other) is service_type(cls)
        and not _HTTP_NAME.search(other.__name__)
        for other in services
    )


def find_service_classes(modules: list) -> list[type]:
    """Find the non-base STT, LLM and TTS classes defined in the given modules."""
    found = []
    for module in modules:
        for cls in vars(module).values():
            if (
                inspect.isclass(cls)
                and cls.__module__ == module.__name__
                and service_type(cls)
                and not is_base_class(cls)
            ):
                found.append(cls)
    return found


def registered_class_names() -> set[str]:
    """Class names listed by every service list in the registry."""
    lists = (
        ServiceRegistry.STT_SERVICES,
        ServiceRegistry.LLM_SERVICES,
        ServiceRegistry.TTS_SERVICES,
        ServiceRegistry.REALTIME_SERVICES,
        ServiceRegistry.VIDEO_SERVICES,
    )
    return {name for services in lists for d in services for name in d.class_name or []}


def main() -> int:
    """Run the check and report every problem found."""
    modules, failures = import_service_modules()
    if failures:
        print("❌ These service modules failed to import, so their services can't be checked:")
        for failure in failures:
            print(f"   - {failure}")
        print()
        print("   Install every extra with")
        print("   `uv sync --group dev --all-extras --no-extra gstreamer --no-extra local`.")
        return 1

    classes = find_service_classes(modules)
    current = [c for c in classes if not is_deprecated(c)]
    concrete = [c for c in current if not is_http_variant(c, current)]
    registered = registered_class_names()

    problems = False

    unlisted = [c for c in concrete if c.__name__ not in registered | EXCLUDED_SERVICES.keys()]
    if unlisted:
        problems = True
        print("❌ Services the CLI neither offers nor excludes:")
        for cls in sorted(unlisted, key=lambda c: c.__name__):
            print(f"   - {cls.__name__} ({cls.__module__})")
        print()
        print("   Add each to src/pipecat/cli/registry/service_metadata.py, or to")
        print("   EXCLUDED_SERVICES there with the reason the CLI leaves it out.")
        print()

    deprecated = [c for c in classes if is_deprecated(c) and c.__name__ in registered]
    if deprecated:
        problems = True
        print("❌ Deprecated services the CLI still offers:")
        for cls in sorted(deprecated, key=lambda c: c.__name__):
            print(f"   - {cls.__name__} ({cls.__module__})")
        print()

    stale = sorted(EXCLUDED_SERVICES.keys() - {c.__name__ for c in concrete})
    if stale:
        problems = True
        print("❌ EXCLUDED_SERVICES names classes that aren't concrete services:")
        for name in stale:
            print(f"   - {name}")
        print()

    if problems:
        return 1
    print(f"✅ All {len(concrete)} service classes are offered or excluded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
