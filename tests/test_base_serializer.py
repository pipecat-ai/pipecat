#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    Frame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TextFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class _Serializer(FrameSerializer):
    """Minimal concrete serializer, so the base class behaviour can be tested."""

    async def serialize(self, frame: Frame) -> str | bytes | None:
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        return None


class TestShouldIgnoreFrame(unittest.TestCase):
    def setUp(self):
        self.serializer = _Serializer()

    def test_ignores_rtvi_message(self):
        """An RTVI-labelled transport message is ignored."""
        for frame_type in (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame):
            with self.subTest(frame_type=frame_type.__name__):
                frame = frame_type(message={"label": RTVI.MESSAGE_LABEL})
                self.assertTrue(self.serializer.should_ignore_frame(frame))

    def test_keeps_other_dict_message(self):
        """A transport message that isn't RTVI is kept."""
        frame = OutputTransportMessageFrame(message={"label": "something-else"})
        self.assertFalse(self.serializer.should_ignore_frame(frame))

    def test_keeps_non_message_frame(self):
        """A frame that isn't a transport message is kept."""
        self.assertFalse(self.serializer.should_ignore_frame(TextFrame(text="hello")))

    def test_non_dict_message_is_kept_and_does_not_raise(self):
        """`message` is typed `Any`, so a non-dict payload must not raise."""
        for payload in ([], ["a"], "str", 123, 1.5, True, None, (1, 2), object()):
            for frame_type in (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame):
                with self.subTest(payload=repr(payload), frame_type=frame_type.__name__):
                    frame = frame_type(message=payload)
                    self.assertFalse(self.serializer.should_ignore_frame(frame))

    def test_ignore_rtvi_messages_disabled(self):
        """With the filter off, RTVI messages are kept."""
        serializer = _Serializer(
            params=FrameSerializer.InputParams(ignore_rtvi_messages=False),
        )
        frame = OutputTransportMessageFrame(message={"label": RTVI.MESSAGE_LABEL})
        self.assertFalse(serializer.should_ignore_frame(frame))


if __name__ == "__main__":
    unittest.main()
