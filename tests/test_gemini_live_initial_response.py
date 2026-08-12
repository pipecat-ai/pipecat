#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test for GeminiLiveLLMService initial-response seeding.

When an initial context has no user/assistant messages and the service is
configured with ``inference_on_context_initialization=True``, the seeding
in ``_handle_context`` must produce a state where
``_create_initial_response`` actually calls ``send_client_content`` on the
session. Previously, the seeded ``role:"system"`` message was extracted
by the Gemini adapter into ``system_instruction``, leaving ``messages``
empty and tripping the ``if not messages`` early-return — so the model
never generated its first response and bots with
``respond_immediately=True`` stayed silent on connect.
"""

from typing import Any

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService


class _FakeSession:
    """Captures every payload sent by the service."""

    def __init__(self):
        self.client_content_calls: list[dict[str, Any]] = []
        self.realtime_input_calls: list[dict[str, Any]] = []

    async def send_client_content(self, **kwargs):
        self.client_content_calls.append(kwargs)

    async def send_realtime_input(self, **kwargs):
        self.realtime_input_calls.append(kwargs)


def _make_service(*, system_instruction: str | None) -> GeminiLiveLLMService:
    """Construct a service with ``inference_on_context_initialization=True``
    wired to a fake session — no I/O."""
    service = GeminiLiveLLMService(
        api_key="test-key",
        system_instruction=system_instruction,
        inference_on_context_initialization=True,
    )
    service._session = _FakeSession()

    async def _noop(*_args, **_kwargs):
        pass

    # Stub out anything that would touch the network or pipeline metrics.
    service._reconnect = _noop  # type: ignore[method-assign]
    service._process_completed_function_calls = _noop  # type: ignore[method-assign]
    service.start_ttfb_metrics = _noop  # type: ignore[method-assign]
    return service


@pytest.mark.asyncio
async def test_handle_context_triggers_initial_response_on_system_only_context():
    """Empty context + ``inference_on_context_initialization=True`` +
    system_instruction present → ``_create_initial_response`` must call
    ``send_client_content`` on the session (i.e. the seed reaches Gemini
    and the model has something to respond to).

    Regression: before this fix, the seeded ``role:"system"`` was
    extracted by the adapter into ``system_instruction`` — ``messages``
    stayed empty, the early-return fired, ``send_client_content`` was
    never called, and the model produced nothing.
    """
    service = _make_service(system_instruction="You are a helpful assistant. Greet the caller.")
    context = LLMContext(messages=[])

    await service._handle_context(context)

    calls = service._session.client_content_calls
    assert calls, (
        "Expected _create_initial_response to invoke send_client_content on the "
        "session so Gemini can produce the initial greeting. Empty call list "
        "means the ``if not messages`` early-return fired — the seeded message "
        "was routed away from ``messages`` by the adapter and Gemini never got "
        "anything to respond to."
    )
    # The seeded turn must be non-empty and terminate with turn_complete=True
    # so Gemini runs inference.
    seed_call = calls[0]
    assert seed_call.get("turns"), "seed must carry at least one turn"
    assert seed_call.get("turn_complete") is True, (
        "turn_complete=True is required to trigger Gemini inference on the seed"
    )


@pytest.mark.asyncio
async def test_handle_context_skips_seeding_when_inference_disabled():
    """``inference_on_context_initialization=False`` → no seed, no call."""
    service = GeminiLiveLLMService(
        api_key="test-key",
        system_instruction="You are a helpful assistant.",
        inference_on_context_initialization=False,
    )
    service._session = _FakeSession()

    async def _noop(*_args, **_kwargs):
        pass

    service._reconnect = _noop  # type: ignore[method-assign]
    service._process_completed_function_calls = _noop  # type: ignore[method-assign]
    service.start_ttfb_metrics = _noop  # type: ignore[method-assign]

    context = LLMContext(messages=[])
    await service._handle_context(context)

    assert service._session.client_content_calls == [], (
        "with inference disabled the service must not send anything on empty context"
    )
