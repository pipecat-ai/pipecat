#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for dynamic RTVIObserver reconfiguration via RTVIConfigureObserverFrame."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
