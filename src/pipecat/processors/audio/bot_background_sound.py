#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from pipecat.audio.utils import resample_audio
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    CancelFrame,
    ControlFrame,
    ErrorFrame,
    OutputAudioRawFrame,
    Frame,
    EndFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

from loguru import logger

try:
    import soundfile as sf
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use background sound, you need to `pip install pipecat-ai[soundfile]`."
    )
    raise Exception(f"Missing module: {e}")


@dataclass
class ChangeBotBackgroundSoundFrame(ControlFrame):
    sound_name: str
    volume: Optional[float] = None


@dataclass
class UpdateBotBackgroundSoundVolumeFrame(ControlFrame):
    volume: float


@dataclass
class PlayBotBackgroundSoundFrame(ControlFrame):
    pass


@dataclass
class PauseBotBackgroundSoundFrame(ControlFrame):
    pass


class BotBackgroundSound(FrameProcessor):
    def __init__(
        self,
        sound_files: Mapping[str, str],
        default_sound: str,
        volume: float = 0.4,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._sound_files = sound_files
        self._volume = volume
        self._sample_rate = sample_rate

        self._sound_pos = 0
        self._sounds: Dict[str, Any] = {}
        self._current_sound = default_sound
        self._playing = True

        self._bot_speaking = False
        self._sleep_time = 0.02

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start()
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStartedFrame):
            self._bot_speaking = True
        elif isinstance(frame, TTSStoppedFrame):
            self._bot_speaking = False
        elif isinstance(frame, TTSAudioRawFrame):
            frame.audio = self._mix_with_sound(frame.audio)
            await self.push_frame(frame)
        elif isinstance(frame, ChangeBotBackgroundSoundFrame):
            await self._change_background_sound(frame)
        elif isinstance(frame, UpdateBotBackgroundSoundVolumeFrame):
            await self._update_background_volume(frame.volume)
        elif isinstance(frame, PlayBotBackgroundSoundFrame):
            await self._play_background_sound(True)
        elif isinstance(frame, PauseBotBackgroundSoundFrame):
            await self._play_background_sound(False)
        else:
            await self.push_frame(frame, direction)

    async def _start(self):
        for sound_name, file_name in self._sound_files.items():
            await asyncio.to_thread(self._load_sound_file, sound_name, file_name)

        self._audio_queue = asyncio.Queue()
        self._audio_task = self.get_event_loop().create_task(self._audio_task_handler())

    async def _stop(self):
        self._audio_task.cancel()
        await self._audio_task

    async def _play_background_sound(self, play: bool):
        self._playing = play

    async def _change_background_sound(self, frame: ChangeBotBackgroundSoundFrame):
        if frame.sound_name in self._sound_files:
            if frame.volume:
                await self._update_background_volume(frame.volume)
            self._current_sound = frame.sound_name
            self._sound_pos = 0
        else:
            error_msg = f"{self} sound {frame.sound_name} is not available"
            logger.error(error_msg)
            await self.push_error(ErrorFrame(error_msg))

    async def _update_background_volume(self, volume: float):
        self._volume = volume

    def _load_sound_file(self, sound_name: str, file_name: str):
        try:
            logger.debug(f"{self} loading background sound from {file_name}")
            sound, sample_rate = sf.read(file_name, dtype="int16")

            audio = sound.tobytes()
            if sample_rate != self._sample_rate:
                logger.debug(f"{self} resampling background sound to {self._sample_rate}")
                audio = resample_audio(audio, sample_rate, self._sample_rate)

            # Convert from np to bytes again.
            self._sounds[sound_name] = np.frombuffer(audio, dtype=np.int16)
        except Exception as ex:
            logger.error(f"{self} unable to open file {file_name}")

    def _mix_with_sound(self, audio: bytes):
        """Mixes raw audio frames with chunks of the same length from the sound
        file.

        """
        if audio:
            audio_np = np.frombuffer(audio, dtype=np.int16)
        else:
            num_samples = int(self._sleep_time * self._sample_rate)
            audio_np = np.zeros(num_samples, dtype=np.int16)

        chunk_size = len(audio_np)

        # Sound currently playing.
        sound = self._sounds[self._current_sound]

        # Go back to the beginning if we don't have enough data.
        if self._sound_pos + chunk_size > len(sound):
            self._sound_pos = 0

        start_pos = self._sound_pos
        end_pos = self._sound_pos + chunk_size
        self._sound_pos = end_pos

        sound_np = sound[start_pos:end_pos]

        mixed_audio = np.clip(audio_np + sound_np * self._volume, -32768, 32767).astype(np.int16)

        return mixed_audio.astype(np.int16).tobytes()

    async def _audio_task_handler(self):
        while True:
            try:
                if self._playing and not self._bot_speaking:
                    audio = self._mix_with_sound(b"")
                    frame = OutputAudioRawFrame(
                        audio=audio, sample_rate=self._sample_rate, num_channels=1
                    )
                    await self.push_frame(frame)
                await asyncio.sleep(self._sleep_time)
            except asyncio.CancelledError:
                break
