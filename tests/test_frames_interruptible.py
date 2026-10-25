#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The interruptible flag on frames: its defaults and how a frame overrides them."""

import unittest
import warnings
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    DataFrame,
    EndFrame,
    FunctionCallResultFrame,
    TextFrame,
    UninterruptibleFrame,
)


class TestInterruptibleFlag(unittest.TestCase):
    def test_defaults_follow_the_class(self):
        self.assertTrue(TextFrame(text="hi").interruptible)
        self.assertFalse(EndFrame().interruptible)
        self.assertFalse(
            FunctionCallResultFrame(
                function_name="f", tool_call_id="1", arguments={}, result={}
            ).interruptible
        )

    def test_a_frame_may_be_set_either_way(self):
        frame = TextFrame(text="hi")
        frame.interruptible = False
        self.assertFalse(frame.interruptible)
        end = EndFrame()
        end.interruptible = True
        self.assertTrue(end.interruptible)

    def test_a_class_declares_its_default_through_the_field(self):
        @dataclass
        class Sticky(DataFrame):
            interruptible: bool = field(default=False, init=False)

        self.assertFalse(Sticky().interruptible)

    def test_the_marker_still_works_and_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            @dataclass
            class Marked(DataFrame, UninterruptibleFrame):
                pass

        self.assertTrue(any(w.category is DeprecationWarning for w in caught))
        self.assertFalse(Marked().interruptible)
