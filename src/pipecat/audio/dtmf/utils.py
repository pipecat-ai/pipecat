#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""DTMF audio utilities.

This module provides functionality to load DTMF (Dual-Tone Multi-Frequency)
audio files corresponding to phone keypad entries. Audio data is cached
in-memory after first load to improve performance on subsequent accesses.
"""

import asyncio
import io
import wave
from importlib.resources import files
from typing import Dict, Optional

import aiofiles

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.utils import create_file_resampler

__DTMF_LOCK__ = asyncio.Lock()
__DTMF_AUDIO__: Dict[KeypadEntry, bytes] = {}
__DTMF_RESAMPLER__: Optional[BaseAudioResampler] = None

__DTMF_FILE_NAME = {
    KeypadEntry.POUND: "dtmf-pound.wav",
    KeypadEntry.STAR: "dtmf-star.wav",
}


async def load_dtmf_audio(button: KeypadEntry, *, sample_rate: int = 8000) -> bytes:
    """Load audio for DTMF tones associated with the given button.

    Args:
        button (KeypadEntry): The button for which the DTMF audio is to be loaded.
        sample_rate (int, optional): The sample rate for the audio. Defaults to 8000.

    Returns:
        bytes: The audio data for the DTMF tone as bytes.
    """
    global __DTMF_AUDIO__, __DTMF_RESAMPLER__

    async with __DTMF_LOCK__:
        if button in __DTMF_AUDIO__:
            return __DTMF_AUDIO__[button]

        if not __DTMF_RESAMPLER__:
            __DTMF_RESAMPLER__ = create_file_resampler()

        dtmf_file_name = __DTMF_FILE_NAME.get(button, f"dtmf-{button.value}.wav")
        dtmf_file_path = files("pipecat.audio.dtmf").joinpath(dtmf_file_name)

        async with aiofiles.open(dtmf_file_path, "rb") as f:
            data = await f.read()

        with io.BytesIO(data) as buffer:
            with wave.open(buffer, "rb") as wf:
                audio = wf.readframes(wf.getnframes())
                in_sample_rate = wf.getframerate()
                resampled_audio = await __DTMF_RESAMPLER__.resample(
                    audio, in_sample_rate, sample_rate
                )
                __DTMF_AUDIO__[button] = resampled_audio

    return __DTMF_AUDIO__[button]
