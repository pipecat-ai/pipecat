#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Vonage Video Connector utils."""

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler


@dataclass
class AudioProps:
    """Audio properties for normalization.

    Parameters:
        sample_rate: The sample rate of the audio.
        is_stereo: Whether the audio is stereo (True) or mono (False).
    """

    sample_rate: int
    is_stereo: bool


class ImageFormat(StrEnum):
    """Enum for image formats."""

    PLANAR_YUV420 = "PLANAR_YUV420"
    PACKED_YUV444 = "PACKED_YUV444"
    RGB = "RGB"
    RGBA = "RGBA"
    BGR = "BGR"
    BGRA = "BGRA"


def check_audio_data(
    buffer: bytes | memoryview, number_of_frames: int, number_of_channels: int
) -> None:
    """Check the audio sample width based on buffer size, number of frames and channels."""
    if number_of_channels not in (1, 2):
        raise ValueError(f"We only accept mono or stereo audio, got {number_of_channels}")

    if isinstance(buffer, memoryview):
        bytes_per_sample = buffer.itemsize
    else:
        bytes_per_sample = len(buffer) // (number_of_frames * number_of_channels)

    if bytes_per_sample != 2:
        raise ValueError(f"We only accept 16 bit PCM audio, got {bytes_per_sample * 8} bit")


def process_audio_channels(
    audio: npt.NDArray[np.int16], current: AudioProps, target: AudioProps
) -> npt.NDArray[np.int16]:
    """Normalize audio channels to the target properties."""
    if current.is_stereo != target.is_stereo:
        if target.is_stereo:
            audio = np.repeat(audio, 2)
        else:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

    return audio


async def process_audio(
    resampler: BaseAudioResampler,
    audio: npt.NDArray[np.int16],
    current: AudioProps,
    target: AudioProps,
) -> npt.NDArray[np.int16]:
    """Normalize audio to the target properties."""
    res_audio = audio
    if current.sample_rate != target.sample_rate:
        # first normalize channels to mono if needed, then resample, then normalize channels to target
        res_audio = process_audio_channels(res_audio, current, replace(current, is_stereo=False))
        current = replace(current, is_stereo=False)
        res_audio_bytes: bytes = await resampler.resample(
            res_audio.tobytes(), current.sample_rate, target.sample_rate
        )
        res_audio = np.frombuffer(res_audio_bytes, dtype=np.int16)

    res_audio = process_audio_channels(res_audio, current, target)

    return res_audio


def image_colorspace_conversion(
    image: bytes, size: tuple[int, int], from_format: ImageFormat, to_format: ImageFormat
) -> bytes | None:
    """Convert image colorspace from one format to another."""
    match (from_format, to_format):
        case (fmt1, fmt2) if fmt1 == fmt2:
            return image
        case (ImageFormat.RGB, ImageFormat.BGR) | (ImageFormat.BGR, ImageFormat.RGB):
            np_input = np.frombuffer(image, dtype=np.uint8)
            np_output = np_input.reshape(size[1], size[0], 3)[:, :, ::-1]
            return np_output.tobytes()
        case (ImageFormat.RGBA, ImageFormat.BGRA) | (ImageFormat.BGRA, ImageFormat.RGBA):
            np_input = np.frombuffer(image, dtype=np.uint8)
            np_output = np_input.reshape(size[1], size[0], 4)[:, :, [2, 1, 0, 3]]
            return np_output.tobytes()
        case (ImageFormat.PLANAR_YUV420, ImageFormat.PACKED_YUV444):
            # YUV420 (I420) has Y plane of size width*height, U and V planes of size (width/2)*(height/2)
            # Packed YUV444 interleaves Y, U, V values for each pixel (YUVYUVYUV...)
            width, height = size
            y_plane_size = width * height
            uv_plane_size_420 = (width // 2) * (height // 2)

            np_input = np.frombuffer(image, dtype=np.uint8)
            y_plane = np_input[:y_plane_size].reshape(height, width)
            u_plane_420 = np_input[y_plane_size : y_plane_size + uv_plane_size_420].reshape(
                height // 2, width // 2
            )
            v_plane_420 = np_input[
                y_plane_size + uv_plane_size_420 : y_plane_size + 2 * uv_plane_size_420
            ].reshape(height // 2, width // 2)

            # Upsample U and V planes by repeating each pixel in 2x2 blocks
            u_plane_444 = np.repeat(np.repeat(u_plane_420, 2, axis=0), 2, axis=1)
            v_plane_444 = np.repeat(np.repeat(v_plane_420, 2, axis=0), 2, axis=1)

            # Interleave Y, U, V values for packed format (YUVYUVYUV...)
            np_output = np.stack([y_plane, u_plane_444, v_plane_444], axis=-1)
            return np_output.tobytes()
        case (ImageFormat.PACKED_YUV444, ImageFormat.PLANAR_YUV420):
            # Packed YUV444 has Y, U, V interleaved (YUVYUVYUV...)
            # YUV420 (I420) has Y plane of size width*height, U and V planes of size (width/2)*(height/2)
            width, height = size

            np_input = np.frombuffer(image, dtype=np.uint8).reshape(height, width, 3)
            y_plane = np_input[:, :, 0].reshape(height, width)
            u_plane_444 = np_input[:, :, 1]
            v_plane_444 = np_input[:, :, 2]

            # Downsample U and V planes by taking every other pixel (2x2 -> 1 averaging)
            u_plane_420 = u_plane_444[::2, ::2].reshape(height // 2, width // 2)
            v_plane_420 = v_plane_444[::2, ::2].reshape(height // 2, width // 2)

            # Concatenate Y, U, V planes
            np_output = np.concatenate(
                [y_plane.flatten(), u_plane_420.flatten(), v_plane_420.flatten()]
            )
            return np_output.tobytes()
        case _:
            return None
