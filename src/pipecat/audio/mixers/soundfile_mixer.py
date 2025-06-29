#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Soundfile-based audio mixer for file playback integration.

Provides an audio mixer that combines incoming audio with audio loaded from
files using the soundfile library. Supports multiple audio formats and
runtime configuration changes.
"""

import asyncio
from typing import Any, Dict, Mapping

import numpy as np
from loguru import logger

from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.frames.frames import MixerControlFrame, MixerEnableFrame, MixerUpdateSettingsFrame

try:
    import soundfile as sf
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the soundfile mixer, you need to `pip install pipecat-ai[soundfile]`."
    )
    raise Exception(f"Missing module: {e}")


class SoundfileMixer(BaseAudioMixer):
    """Audio mixer that combines incoming audio with file-based audio.

    This is an audio mixer that mixes incoming audio with audio from a
    file. It uses the soundfile library to load files so it supports multiple
    formats. The audio files need to only have one channel (mono) and it needs
    to match the sample rate of the output transport.

    Multiple files can be loaded, each with a different name. The
    `MixerUpdateSettingsFrame` has the following settings available: `sound`
    (str) and `volume` (float) to be able to update to a different sound file or
    to change the volume at runtime.
    """

    def __init__(
        self,
        *,
        sound_files: Mapping[str, str],
        default_sound: str,
        volume: float = 0.4,
        mixing: bool = True,
        loop: bool = True,
        **kwargs,
    ):
        """Initialize the soundfile mixer.

        Args:
            sound_files: Mapping of sound names to file paths for loading.
            default_sound: Name of the default sound to play initially.
            volume: Mixing volume level (0.0 to 1.0). Defaults to 0.4.
            mixing: Whether mixing is initially enabled. Defaults to True.
            loop: Whether to loop audio files when they end. Defaults to True.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._sound_files = sound_files
        self._volume = volume
        self._sample_rate = 0

        self._sound_pos = 0
        self._sounds: Dict[str, Any] = {}
        self._current_sound = default_sound
        self._mixing = mixing
        self._loop = loop

    async def start(self, sample_rate: int):
        """Initialize the mixer and load all sound files.

        Args:
            sample_rate: The sample rate of the output transport in Hz.
        """
        self._sample_rate = sample_rate
        for sound_name, file_name in self._sound_files.items():
            await asyncio.to_thread(self._load_sound_file, sound_name, file_name)

    async def stop(self):
        """Clean up mixer resources.

        Currently performs no cleanup as sound data is managed by garbage collection.
        """
        pass

    async def process_frame(self, frame: MixerControlFrame):
        """Process mixer control frames to update settings or enable/disable mixing.

        Args:
            frame: The mixer control frame to process.
        """
        if isinstance(frame, MixerUpdateSettingsFrame):
            await self._update_settings(frame)
        elif isinstance(frame, MixerEnableFrame):
            await self._enable_mixing(frame.enable)
        pass

    async def mix(self, audio: bytes) -> bytes:
        """Mix transport audio with the current sound file.

        Args:
            audio: Raw audio bytes from the transport to mix.

        Returns:
            Mixed audio bytes combining transport and file audio.
        """
        return self._mix_with_sound(audio)

    async def _enable_mixing(self, enable: bool):
        """Enable or disable audio mixing."""
        self._mixing = enable

    async def _update_settings(self, frame: MixerUpdateSettingsFrame):
        """Update mixer settings from a control frame."""
        for setting, value in frame.settings.items():
            match setting:
                case "sound":
                    await self._change_sound(value)
                case "volume":
                    await self._update_volume(value)
                case "loop":
                    await self._update_loop(value)

    async def _change_sound(self, sound: str):
        """Change the currently playing sound file.

        Args:
            sound: Name of the sound file to switch to.
        """
        if sound in self._sound_files:
            self._current_sound = sound
            self._sound_pos = 0
        else:
            logger.error(f"Sound {sound} is not available")

    async def _update_volume(self, volume: float):
        """Update the mixing volume level."""
        self._volume = volume

    async def _update_loop(self, loop: bool):
        """Update the looping behavior."""
        self._loop = loop

    def _load_sound_file(self, sound_name: str, file_name: str):
        """Load an audio file into memory for mixing."""
        try:
            logger.debug(f"Loading mixer sound from {file_name}")
            sound, sample_rate = sf.read(file_name, dtype="int16")

            if sample_rate == self._sample_rate:
                audio = sound.tobytes()
                # Convert from np to bytes again.
                self._sounds[sound_name] = np.frombuffer(audio, dtype=np.int16)
            else:
                logger.warning(
                    f"Sound file {file_name} has incorrect sample rate {sample_rate} (should be {self._sample_rate})"
                )
        except Exception as e:
            logger.error(f"Unable to open file {file_name}: {e}")

    def _mix_with_sound(self, audio: bytes):
        """Mix raw audio frames with chunks of the same length from the sound file."""
        if not self._mixing or not self._current_sound in self._sounds:
            return audio

        audio_np = np.frombuffer(audio, dtype=np.int16)
        chunk_size = len(audio_np)

        # Sound currently playing.
        sound = self._sounds[self._current_sound]

        # Go back to the beginning if we don't have enough data.
        if self._sound_pos + chunk_size > len(sound):
            if not self._loop:
                return audio
            self._sound_pos = 0

        start_pos = self._sound_pos
        end_pos = self._sound_pos + chunk_size
        self._sound_pos = end_pos

        sound_np = sound[start_pos:end_pos]

        mixed_audio = np.clip(audio_np + sound_np * self._volume, -32768, 32767).astype(np.int16)

        return mixed_audio.astype(np.int16).tobytes()
