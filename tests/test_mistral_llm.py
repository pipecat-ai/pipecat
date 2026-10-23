#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the Mistral LLM service's request parameters."""

from unittest.mock import patch

import pytest
from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN

from pipecat.adapters.llm.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.mistral.llm import MistralLLMService


def _build_params(settings=None):
    with patch.object(MistralLLMService, "create_client"):
        service = MistralLLMService(api_key="test-key", settings=settings)
    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Hello"}],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )
    return service.build_chat_completion_params(invocation)


@pytest.mark.parametrize("seed", [0, 42])
def test_seed_is_sent_as_random_seed(seed):
    params = _build_params(MistralLLMService.Settings(seed=seed))

    assert params["random_seed"] == seed


def test_unset_seed_is_not_sent():
    params = _build_params()

    assert "random_seed" not in params
