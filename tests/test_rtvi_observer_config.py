#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for dynamic RTVIObserver reconfiguration via RTVIConfigureObserverFrame."""

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import LLMMarkerFrame, LLMMarkerResponseFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIConfigureObserverFrame
from pipecat.processors.frameworks.rtvi.observer import (
    RTVIFunctionCallReportLevel,
    RTVIObserver,
    RTVIObserverParams,
)


class TestRTVIConfigureObserver(unittest.TestCase):
    def test_raises_report_level_at_runtime(self):
        # Agents default to the secure NONE; a config frame elevates it live.
        observer = RTVIObserver(params=RTVIObserverParams())
        self.assertEqual(
            observer._get_function_call_report_level("get_weather"),
            RTVIFunctionCallReportLevel.NONE,
        )
        observer._apply_config(
            RTVIConfigureObserverFrame(
                function_call_report_level={"*": RTVIFunctionCallReportLevel.FULL}
            )
        )
        self.assertEqual(
            observer._get_function_call_report_level("get_weather"),
            RTVIFunctionCallReportLevel.FULL,
        )

    def test_none_field_leaves_config_unchanged(self):
        observer = RTVIObserver(
            params=RTVIObserverParams(
                function_call_report_level={"*": RTVIFunctionCallReportLevel.NAME}
            )
        )
        observer._apply_config(RTVIConfigureObserverFrame(function_call_report_level=None))
        self.assertEqual(
            observer._get_function_call_report_level("get_weather"),
            RTVIFunctionCallReportLevel.NAME,
        )

    def test_enables_vad_user_speaking_at_runtime(self):
        # Off by default; a config frame enables raw VAD speaking events live.
        observer = RTVIObserver(params=RTVIObserverParams())
        self.assertFalse(observer._params.vad_user_speaking_enabled)
        observer._apply_config(RTVIConfigureObserverFrame(vad_user_speaking_enabled=True))
        self.assertTrue(observer._params.vad_user_speaking_enabled)
        # A None field leaves it unchanged.
        observer._apply_config(RTVIConfigureObserverFrame(vad_user_speaking_enabled=None))
        self.assertTrue(observer._params.vad_user_speaking_enabled)

    def test_enables_llm_markers_at_runtime(self):
        # Off by default; a config frame enables the LLM's sideband markers live.
        observer = RTVIObserver(params=RTVIObserverParams())
        self.assertFalse(observer._params.bot_llm_marker_enabled)
        observer._apply_config(RTVIConfigureObserverFrame(bot_llm_marker_enabled=True))
        self.assertTrue(observer._params.bot_llm_marker_enabled)
        observer._apply_config(RTVIConfigureObserverFrame(bot_llm_marker_enabled=None))
        self.assertTrue(observer._params.bot_llm_marker_enabled)


class TestRTVIObserverLLMMarkers(unittest.IsolatedAsyncioTestCase):
    async def _push(self, observer: RTVIObserver, frame) -> list:
        sent = []
        observer.send_rtvi_message = AsyncMock(side_effect=lambda m: sent.append(m))
        source = FrameProcessor()
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=source,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        return sent

    async def test_markers_are_not_sent_by_default(self):
        observer = RTVIObserver(params=RTVIObserverParams())
        sent = await self._push(observer, LLMMarkerResponseFrame(raw="● Hi", marker="●"))
        self.assertEqual(sent, [])

    async def test_marker_is_sent_when_enabled(self):
        observer = RTVIObserver(params=RTVIObserverParams(bot_llm_marker_enabled=True))
        frame = LLMMarkerResponseFrame(
            raw="● Hi there", marker="●", kind="complete", markers=["●", "◐", "○"]
        )
        sent = await self._push(observer, frame)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].type, "bot-llm-marker")
        self.assertEqual(sent[0].data.text, "●")
        self.assertEqual(sent[0].data.kind, "complete")
        self.assertEqual(sent[0].data.raw, "● Hi there")
        self.assertEqual(sent[0].data.markers, ["●", "◐", "○"])

    async def test_a_response_without_a_marker_is_reported_too(self):
        observer = RTVIObserver(params=RTVIObserverParams(bot_llm_marker_enabled=True))
        sent = await self._push(observer, LLMMarkerResponseFrame(raw="Hi there"))
        self.assertEqual((sent[0].data.text, sent[0].data.kind), ("", None))

    async def test_the_marker_frame_itself_is_not_sent(self):
        # The context aggregator's marker frame is not the report; the report
        # comes once per response, when it ends.
        observer = RTVIObserver(params=RTVIObserverParams(bot_llm_marker_enabled=True))
        sent = await self._push(observer, LLMMarkerFrame(marker="●", kind="complete"))
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
