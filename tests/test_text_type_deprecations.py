#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the deprecated names of ``text_type``."""

import importlib
from abc import ABC
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pytest

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    Frame,
    TTSTextFrame,
)
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_aggregator import TextType


class _TestTTSService(TTSService):
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield  # pragma: no cover


@dataclass
class _AppTextFrame(TTSTextFrame):
    extra: int = 0


class _AbstractAppTextFrame(_AppTextFrame, ABC):
    pass


async def _transform(text: str, text_type: str) -> str:
    return text


async def _bot_output_transform(text, text_type, accumulated_text=None, remaining_text=None):
    return RTVI.BotOutputTransformResult(text=text)


@pytest.mark.parametrize("frame_cls", [AggregatedTextFrame, TTSTextFrame])
def test_aggregated_by_keyword_sets_text_type(frame_cls):
    with pytest.warns(DeprecationWarning, match="is deprecated since 1.13.0"):
        frame = frame_cls("Hello.", aggregated_by=TextType.WORD)
    assert frame.text_type == TextType.WORD


@pytest.mark.parametrize("frame_cls", [_AppTextFrame, _AbstractAppTextFrame])
def test_aggregated_by_keyword_on_a_subclass(frame_cls):
    with pytest.warns(DeprecationWarning, match="`AggregatedTextFrame.aggregated_by`"):
        frame = frame_cls("Hello.", aggregated_by="status", extra=1)
    assert (frame.text_type, frame.extra) == ("status", 1)


def test_text_type_by_position():
    frame = TTSTextFrame("Hello.", TextType.WORD)
    assert frame.text_type == TextType.WORD


def test_reading_aggregated_by_returns_text_type():
    frame = AggregatedTextFrame("Hello.", text_type="status")
    with pytest.warns(DeprecationWarning, match="`AggregatedTextFrame.aggregated_by`"):
        assert frame.aggregated_by == "status"


def test_assigning_aggregated_by_sets_text_type():
    frame = TTSTextFrame("Hello.", text_type=TextType.SENTENCE)
    with pytest.warns(DeprecationWarning, match="`AggregatedTextFrame.aggregated_by`"):
        frame.aggregated_by = "status"
    assert frame.text_type == "status"


def test_progress_frame_aggregated_by():
    with pytest.warns(DeprecationWarning, match="`AggregatedTextProgressFrame"):
        frame = AggregatedTextProgressFrame(
            segment_id=1,
            context_id="ctx",
            text="Hello there.",
            aggregated_by="status",
            accumulated_text="Hello",
            remaining_text=" there.",
        )
    assert frame.text_type == "status"
    with pytest.warns(DeprecationWarning, match="`AggregatedTextProgressFrame.aggregated_by`"):
        frame.aggregated_by = TextType.SENTENCE
    assert frame.text_type == TextType.SENTENCE


def test_rtvi_observer_params_skip_aggregator_types():
    with pytest.warns(DeprecationWarning, match="`RTVIObserverParams.skip_aggregator_types`"):
        params = RTVIObserverParams(skip_aggregator_types=["status"])
    assert params.skip_text_types == ["status"]


def test_rtvi_observer_transformer_aggregation_type():
    observer = RTVIObserver()
    with pytest.warns(DeprecationWarning, match="`aggregation_type` is deprecated"):
        observer.add_bot_output_transformer(_bot_output_transform, aggregation_type="status")
    assert [t[0] for t in observer._aggregation_transforms] == ["status"]
    with pytest.warns(DeprecationWarning, match="`aggregation_type` is deprecated"):
        observer.remove_bot_output_transformer(_bot_output_transform, aggregation_type="status")
    assert observer._aggregation_transforms == []


def test_tts_service_skip_aggregator_types():
    with pytest.warns(DeprecationWarning, match="`skip_aggregator_types` is deprecated"):
        tts = _TestTTSService(skip_aggregator_types=["status"])
    assert tts._skip_text_types == ["status"]


def test_tts_service_transformer_aggregation_type():
    tts = _TestTTSService()
    with pytest.warns(DeprecationWarning, match="`aggregation_type` is deprecated"):
        tts.add_text_transformer(_transform, aggregation_type="status")
    assert tts._text_transforms == [("status", _transform)]
    with pytest.warns(DeprecationWarning, match="`aggregation_type` is deprecated"):
        tts.remove_text_transformer(_transform, aggregation_type="status")
    assert tts._text_transforms == []


@pytest.mark.parametrize("field", ["text_type", "aggregated_by"])
def test_bot_output_message_carries_both_fields(field):
    data = RTVI.BotOutputMessageData(text="Hello.", **{field: "status"})
    assert data.model_dump(exclude_none=True) == {
        "text": "Hello.",
        "text_type": "status",
        "aggregated_by": "status",
    }


@pytest.mark.parametrize(
    "module", ["pipecat.utils.text.base_text_aggregator", "pipecat.frames.frames"]
)
def test_aggregation_type_is_text_type(module):
    with pytest.warns(DeprecationWarning, match="`AggregationType` is deprecated"):
        aggregation_type = importlib.import_module(module).AggregationType
    assert aggregation_type is TextType
