#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The frame queue reads a frame's ``interruptible`` flag as it is."""

import unittest

from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.utils.frame_queue import FrameQueue


class TestFrameQueueInterruptibility(unittest.IsolatedAsyncioTestCase):
    def test_a_flag_set_on_a_queued_frame_is_respected(self):
        queue = FrameQueue()
        frame = TextFrame(text="hi")
        queue.put_nowait(frame)
        self.assertFalse(queue.has_uninterruptible)

        frame.interruptible = False
        self.assertTrue(queue.has_uninterruptible)
        queue.get_nowait()
        self.assertFalse(queue.has_uninterruptible)

        end = EndFrame()
        queue.put_nowait(end)
        self.assertTrue(queue.has_uninterruptible)
        end.interruptible = True
        self.assertFalse(queue.has_uninterruptible)

    def test_reset_keeps_what_is_uninterruptible_now(self):
        queue = FrameQueue()
        plain, end = TextFrame(text="hi"), EndFrame()
        queue.put_nowait(plain)
        queue.put_nowait(end)
        plain.interruptible = False
        end.interruptible = True

        queue.reset()

        self.assertEqual(queue.qsize(), 1)
        self.assertIs(queue.get_nowait(), plain)
        self.assertFalse(queue.has_uninterruptible)

    def test_has_frame_and_tuple_items(self):
        queue = FrameQueue(frame_getter=lambda item: item[0])
        queue.put_nowait((TextFrame(text="hi"), "down"))
        queue.put_nowait((EndFrame(), "down"))
        self.assertTrue(queue.has_frame(TextFrame))
        self.assertTrue(queue.has_uninterruptible)
        queue.reset()
        self.assertFalse(queue.has_frame(TextFrame))
        self.assertTrue(queue.has_frame(EndFrame))
        self.assertEqual(queue.get_nowait()[1], "down")
