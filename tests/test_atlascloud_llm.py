#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import patch

from pipecat.cli.registry import ServiceLoader, ServiceRegistry
from pipecat.services.atlascloud.llm import AtlasCloudLLMService


def test_atlascloud_llm_defaults_to_openai_compatible_endpoint():
    with patch.object(AtlasCloudLLMService, "create_client") as create_client:
        service = AtlasCloudLLMService(api_key="test-key")

    assert service._settings.model == "qwen/qwen3.5-flash"
    assert service.supports_developer_role is False
    create_client.assert_called_once_with(
        api_key="test-key",
        base_url="https://api.atlascloud.ai/v1",
        organization=None,
        project=None,
        default_headers=None,
    )


def test_atlascloud_llm_registry_metadata_and_generated_config():
    service = ServiceLoader.get_service_by_value(
        ServiceRegistry.LLM_SERVICES,
        "atlascloud_llm",
    )

    assert service is not None
    assert service.label == "Atlas Cloud"
    assert service.package == "pipecat-ai[atlascloud]"
    assert ServiceLoader.get_service_import("atlascloud_llm") == [
        "from pipecat.services.atlascloud.llm import AtlasCloudLLMService"
    ]
    config = ServiceLoader.get_service_config("atlascloud_llm")
    assert config is not None
    assert "ATLASCLOUD_API_KEY" in config
    assert 'model=os.getenv("ATLASCLOUD_MODEL", "qwen/qwen3.5-flash")' in config
