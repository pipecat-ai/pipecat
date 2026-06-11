#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Verify the deprecated task frame aliases warn and stay isinstance-compatible."""

import unittest
import warnings

from pipecat.frames.frames import (
    CancelTaskFrame,
    CancelWorkerFrame,
    EndTaskFrame,
    EndWorkerFrame,
    InterruptionTaskFrame,
    InterruptionWorkerFrame,
    StopTaskFrame,
    StopWorkerFrame,
    TaskFrame,
    TaskSystemFrame,
    WorkerFrame,
    WorkerSystemFrame,
)

ALIASES = [
    (EndTaskFrame, EndWorkerFrame, TaskFrame),
    (StopTaskFrame, StopWorkerFrame, TaskFrame),
    (CancelTaskFrame, CancelWorkerFrame, TaskSystemFrame),
    (InterruptionTaskFrame, InterruptionWorkerFrame, TaskSystemFrame),
]


class TestTaskFrameAliases(unittest.TestCase):
    def test_aliases_warn_and_remain_isinstance_compatible(self):
        for old_cls, new_cls, old_base in ALIASES:
            with self.subTest(alias=old_cls.__name__):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    frame = old_cls()

                deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
                self.assertEqual(len(deprecations), 1)
                self.assertIn(f"{old_cls.__name__} is deprecated", str(deprecations[0].message))
                self.assertIn(new_cls.__name__, str(deprecations[0].message))

                self.assertIsInstance(frame, new_cls)
                self.assertIsInstance(frame, old_base)

    def test_new_frames_do_not_warn(self):
        for new_cls in (
            EndWorkerFrame,
            StopWorkerFrame,
            CancelWorkerFrame,
            InterruptionWorkerFrame,
        ):
            with self.subTest(frame=new_cls.__name__):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    frame = new_cls()

                deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
                self.assertEqual(len(deprecations), 0)
                self.assertNotIsInstance(frame, (TaskFrame, TaskSystemFrame))

    def test_reason_is_preserved(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(EndTaskFrame(reason="done").reason, "done")
            self.assertEqual(CancelTaskFrame(reason="bye").reason, "bye")

    def test_base_aliases_subclass_worker_frames(self):
        self.assertTrue(issubclass(TaskFrame, WorkerFrame))
        self.assertTrue(issubclass(TaskSystemFrame, WorkerSystemFrame))


if __name__ == "__main__":
    unittest.main()
