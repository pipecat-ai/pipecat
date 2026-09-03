#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AWS Transcribe STT language code mapping."""

import pytest

from pipecat.services.aws.stt import AWSTranscribeSTTService
from pipecat.transcriptions.language import Language


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        (Language.CKB_IQ, "ckb-IQ"),
        (Language.CKB_IR, "ckb-IR"),
        (Language.EN_AB, "en-AB"),
        (Language.EN_WL, "en-WL"),
        (Language.FA_AF, "fa-AF"),
        (Language.KAB, "kab-DZ"),
        (Language.KAB_DZ, "kab-DZ"),
        (Language.SW_BI, "sw-BI"),
        (Language.SW_RW, "sw-RW"),
        (Language.SW_UG, "sw-UG"),
    ],
)
def test_aws_streaming_languages(language: Language, expected: str):
    """AWS streaming language variants should resolve to their provider codes."""
    service = object.__new__(AWSTranscribeSTTService)

    assert service.language_to_service_language(language) == expected
