#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how LLM services log the context in each ``LLMContextLogMode``."""

import io
import sys

import pytest
from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService
from pipecat.utils import log_config
from pipecat.utils.log_config import LLM_CONTEXT_LOG_ENV_VAR, configure_logging

MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What's the weather?"},
]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(log_config, "_config", log_config._LogConfig())
    monkeypatch.delenv(LLM_CONTEXT_LOG_ENV_VAR, raising=False)


@pytest.fixture
def captured():
    sink = io.StringIO()
    handler_id = logger.add(sink, level="DEBUG", format="{name}|{message}")
    yield sink
    logger.remove(handler_id)


def test_full_logs_every_message(captured):
    service = LLMService()
    service._log_llm_context(LLMContext(messages=MESSAGES))
    assert captured.getvalue() == (
        f"{__name__}|{service}: Generating chat from context {MESSAGES}\n"
    )


def test_conversation_setup_logs_every_message(captured):
    service = LLMService()
    service._log_llm_conversation_setup(LLMContext(messages=MESSAGES))
    assert captured.getvalue() == (
        f"{__name__}|{service}: Setting up conversation from context {MESSAGES}\n"
    )


def test_off_logs_nothing(captured):
    configure_logging(llm_context="off")
    service = LLMService()
    service._log_llm_context(LLMContext(messages=MESSAGES))
    service._log_llm_conversation_setup(LLMContext(messages=MESSAGES))
    assert captured.getvalue() == ""


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
        service._log_llm_context(LLMContext(messages=MESSAGES))
    finally:
        # Put loguru's default sink back so the rest of the session still logs.
        logger.remove()
        logger.add(sys.stderr)
    assert rendered == []
