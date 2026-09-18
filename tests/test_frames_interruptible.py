#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The interruptible flag on frames: its defaults and how a frame overrides them."""

import unittest

from pipecat.frames.frames import EndFrame, FunctionCallResultFrame, TextFrame


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
