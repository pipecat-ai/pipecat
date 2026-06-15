#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import patch

from pipecat.services.huggingface.llm import (
    HuggingFaceLLMService,
    HuggingFaceLLMSettings,
)


def test_huggingface_llm_uses_router_headers():
    with patch.object(
        HuggingFaceLLMService, "create_client", return_value=object()
    ) as create_client:
        service = HuggingFaceLLMService(
            api_key="hf_test",
            bill_to="demo-org",
            settings=HuggingFaceLLMSettings(model="deepseek-ai/DeepSeek-R1:fastest"),
        )

    assert service.supports_developer_role is False
    assert service._settings.model == "deepseek-ai/DeepSeek-R1:fastest"
    kwargs = create_client.call_args.kwargs
    assert kwargs["api_key"] == "hf_test"
    assert kwargs["base_url"] == "https://router.huggingface.co/v1"
    assert kwargs["default_headers"] == {"X-HF-Bill-To": "demo-org"}
