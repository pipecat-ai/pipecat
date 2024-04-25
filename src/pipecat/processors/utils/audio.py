#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List

from pipecat.frames.frames import AudioRawFrame


def maybe_split_audio_frame(frame: AudioRawFrame, largest_write_size: int) -> List[AudioRawFrame]:
    """Subdivide large audio frames to enable interruption."""
    frames: List[AudioRawFrame] = []
    if len(frame.data) > largest_write_size:
        for i in range(0, len(frame.data), largest_write_size):
            chunk = frame.data[i: i + largest_write_size]
            frames.append(AudioRawFrame(chunk, frame.sample_rate, frame.num_channels))
    else:
        frames.append(frame)
    return frames
