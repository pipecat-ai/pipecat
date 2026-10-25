#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how LLM services log the context in each ``PIPECAT_LOG_LLM_CONTEXT`` mode."""

import io
import sys

import pytest
from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService
from pipecat.utils import log_config
from pipecat.utils.env import InvalidEnvVarValueError
from pipecat.utils.log_config import _LLM_CONTEXT_LOG_ENV_VAR

MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What's the weather?"},
]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(log_config, "_llm_context_log_mode", None)
    monkeypatch.delenv(_LLM_CONTEXT_LOG_ENV_VAR, raising=False)


@pytest.fixture
def captured():
    sink = io.StringIO()
    handler_id = logger.add(sink, level="DEBUG", format="{name}|{message}")
    yield sink
    logger.remove(handler_id)


def test_invalid_mode_raises_at_construction(monkeypatch):
    monkeypatch.setenv(_LLM_CONTEXT_LOG_ENV_VAR, "of")
    with pytest.raises(InvalidEnvVarValueError):
        LLMService()


def test_full_logs_every_message(captured):
    service = LLMService()
    service._log_llm_response(LLMContext(messages=MESSAGES))
    assert captured.getvalue() == (f"{__name__}|{service}: Generating LLM response {MESSAGES}\n")


def test_conversation_setup_logs_every_message(captured):
    service = LLMService()
    service._log_llm_conversation_setup(LLMContext(messages=MESSAGES))
    assert captured.getvalue() == (
        f"{__name__}|{service}: Setting up LLM conversation {MESSAGES}\n"
    )


def test_off_logs_without_context(captured, monkeypatch):
    monkeypatch.setenv(_LLM_CONTEXT_LOG_ENV_VAR, "off")
    service = LLMService()
    service._log_llm_response(LLMContext(messages=MESSAGES))
    service._log_llm_conversation_setup(LLMContext(messages=MESSAGES))
    assert captured.getvalue().splitlines() == [
        f"{__name__}|{service}: Generating LLM response",
        f"{__name__}|{service}: Setting up LLM conversation",
    ]


def test_off_does_not_build_messages(captured, monkeypatch):
    monkeypatch.setenv(_LLM_CONTEXT_LOG_ENV_VAR, "off")
    service = LLMService()
    rendered = []
    monkeypatch.setattr(
        service.get_llm_adapter(),
        "get_messages_for_logging",
        lambda context: rendered.append(context) or [],
    )
    service._log_llm_response(LLMContext(messages=MESSAGES))
    assert rendered == []


def test_not_rendered_without_debug_sink(monkeypatch):
    service = LLMService()
    rendered = []
    monkeypatch.setattr(
        service.get_llm_adapter(),
        "get_messages_for_logging",
        lambda context: rendered.append(context) or [],
    )
    logger.remove()
    try:
        logger.add(io.StringIO(), level="INFO")
        service._log_llm_response(LLMContext(messages=MESSAGES))
    finally:
        # Put loguru's default sink back so the rest of the session still logs.
        logger.remove()
        logger.add(sys.stderr)
    assert rendered == []
