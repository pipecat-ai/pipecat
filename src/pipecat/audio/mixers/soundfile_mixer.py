#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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
    """This is an audio mixer that mixes incoming audio with audio from a
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
        self._sample_rate = sample_rate
        for sound_name, file_name in self._sound_files.items():
            await asyncio.to_thread(self._load_sound_file, sound_name, file_name)

    async def stop(self):
        pass

    async def process_frame(self, frame: MixerControlFrame):
        if isinstance(frame, MixerUpdateSettingsFrame):
            await self._update_settings(frame)
        elif isinstance(frame, MixerEnableFrame):
            await self._enable_mixing(frame.enable)
        pass

    async def mix(self, audio: bytes) -> bytes:
        return self._mix_with_sound(audio)

    async def _enable_mixing(self, enable: bool):
        self._mixing = enable

    async def _update_settings(self, frame: MixerUpdateSettingsFrame):
        for setting, value in frame.settings.items():
            match setting:
                case "sound":
                    await self._change_sound(value)
                case "volume":
                    await self._update_volume(value)
                case "loop":
                    await self._update_loop(value)

    async def _change_sound(self, sound: str):
        if sound in self._sound_files:
            self._current_sound = sound
            self._sound_pos = 0
        else:
            logger.error(f"Sound {sound} is not available")

    async def _update_volume(self, volume: float):
        self._volume = volume

    async def _update_loop(self, loop: bool):
        self._loop = loop

    def _load_sound_file(self, sound_name: str, file_name: str):
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
        """Mixes raw audio frames with chunks of the same length from the sound
        file.

        """
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
