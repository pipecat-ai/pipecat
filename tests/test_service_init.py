#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for service settings and initialization patterns.

Settings objects operate in two modes:

- **Store mode** (``self._settings``): the live state inside a service.
  Every field must hold a real value (``None`` is fine, ``NOT_GIVEN`` is not).
- **Delta mode** (``FooSettings()`` with no args): a sparse update.
  Every field must default to ``NOT_GIVEN`` so ``apply_update()`` skips
  untouched fields and doesn't accidentally overwrite the store.

These tests verify both sides of that contract automatically:

1. **Delta defaults** — Instantiate every ``ServiceSettings`` subclass with
   no arguments and assert that every field is ``NOT_GIVEN``.  Catches the
   bug where a field defaults to ``None`` instead of ``NOT_GIVEN``, which
   would cause partial deltas to silently overwrite unrelated store values.

2. **Store completeness** — Instantiate every concrete service with dummy
   args and assert that ``_settings`` contains no ``NOT_GIVEN`` values.
   This is the same check that ``validate_complete()`` runs in ``start()``,
   but caught here at unit-test time without needing a running pipeline.
   Catches services that forget to initialize a field in ``default_settings``.

All Settings and Service classes are auto-discovered via ``pkgutil``;
new services are covered automatically with no per-service maintenance.
"""

import importlib
import inspect
import pkgutil
from dataclasses import fields

import pytest

import pipecat.services
from pipecat.services.ai_service import AIService
from pipecat.services.settings import ServiceSettings, is_given

# Modules that define abstract base service classes (not concrete services).
_BASE_MODULES = frozenset(
    {
        "pipecat.services.ai_service",
        "pipecat.services.llm_service",
        "pipecat.services.stt_service",
        "pipecat.services.tts_service",
        "pipecat.services.image_gen_service",
        "pipecat.services.vision_service",
    }
)


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------


def _all_subclasses(cls):
    result = set()
    for sub in cls.__subclasses__():
        result.add(sub)
        result.update(_all_subclasses(sub))
    return result


def _import_all_service_modules():
    """Import every module under pipecat.services (skipping missing deps)."""
    package = pipecat.services
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        package.__path__, prefix=package.__name__ + ".", onerror=lambda _name: None
    ):
        try:
            importlib.import_module(modname)
        except Exception:
            continue


_import_all_service_modules()

ALL_SETTINGS_CLASSES = sorted(_all_subclasses(ServiceSettings), key=lambda c: c.__qualname__)
assert ALL_SETTINGS_CLASSES, "No settings classes discovered"


# ---------------------------------------------------------------------------
# Service instantiation helpers
# ---------------------------------------------------------------------------


def _try_instantiate(cls):
    """Try to instantiate a service with dummy values for required args.

    Inspects the __init__ signature and passes "test" for every required
    keyword-only parameter.  Services that need non-string required args
    or fail for other reasons will raise and be skipped by the test.
    """
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is not param.empty:
            continue
        # Required parameter — pass a dummy string
        kwargs[name] = "test"
    return cls(**kwargs)


def _discover_service_classes():
    """Return concrete service classes that can be instantiated with dummy args."""
    result = []
    for cls in sorted(_all_subclasses(AIService), key=lambda c: c.__qualname__):
        # Skip abstract base classes defined in framework modules.
        if cls.__module__ in _BASE_MODULES:
            continue
        try:
            svc = _try_instantiate(cls)
        except Exception:
            continue
        if hasattr(svc, "_settings"):
            result.append(cls)
    return result


ALL_SERVICE_CLASSES = _discover_service_classes()
assert ALL_SERVICE_CLASSES, "No service classes could be instantiated"


# ---------------------------------------------------------------------------
# 1. Settings defaults: delta-mode safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("settings_cls", ALL_SETTINGS_CLASSES, ids=lambda c: c.__qualname__)
def test_delta_defaults_are_not_given(settings_cls):
    """Every field must default to NOT_GIVEN so empty deltas are no-ops.

    A field that defaults to None instead of NOT_GIVEN will cause
    apply_update() to overwrite the corresponding store value whenever
    a partial delta is applied.
    """
    instance = settings_cls()
    for f in fields(instance):
        if f.name == "extra":
            continue
        val = getattr(instance, f.name)
        assert not is_given(val), (
            f"{settings_cls.__qualname__}.{f.name} defaults to {val!r}, expected NOT_GIVEN"
        )


# ---------------------------------------------------------------------------
# 2. Service construction: store-mode completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("service_cls", ALL_SERVICE_CLASSES, ids=lambda c: c.__qualname__)
def test_service_settings_complete(service_cls):
    """After construction, _settings must have no NOT_GIVEN values.

    This is what validate_complete() checks in start().  Catching it
    here means we don't need a running pipeline to find missing defaults.
    """
    try:
        svc = _try_instantiate(service_cls)
    except Exception:
        pytest.skip("Cannot re-instantiate (environment issue)")
    for f in fields(svc._settings):
        if f.name == "extra":
            continue
        val = getattr(svc._settings, f.name)
        assert is_given(val), (
            f"{service_cls.__qualname__}._settings.{f.name} is NOT_GIVEN after construction"
        )
