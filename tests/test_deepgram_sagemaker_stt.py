#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the Deepgram SageMaker STT service's query string."""

from urllib.parse import parse_qs

import pytest

pytest.importorskip("pipecat.services.deepgram.sagemaker.stt", exc_type=ImportError)

from pipecat.services.deepgram.sagemaker.stt import DeepgramSageMakerSTTService  # noqa: E402


def _query(settings: DeepgramSageMakerSTTService.Settings) -> dict[str, list[str]]:
    service = DeepgramSageMakerSTTService(
        endpoint_name="endpoint", region="us-east-1", sample_rate=16000, settings=settings
    )
    return parse_qs(service._build_query_string())


def test_list_settings_become_one_parameter_per_item():
    query = _query(DeepgramSageMakerSTTService.Settings(keyterm=["pipecat", "daily"]))

    assert query["keyterm"] == ["pipecat", "daily"]


def test_values_are_url_encoded():
    query = _query(
        DeepgramSageMakerSTTService.Settings(
            keyterm=["Q&A session"], extra={"tag": "a:b c", "flag": True}
        )
    )

    assert query["keyterm"] == ["Q&A session"]
    assert query["tag"] == ["a:b c"]
    assert query["flag"] == ["true"]
