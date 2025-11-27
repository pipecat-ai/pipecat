import asyncio
import os
import time
import wave

from loguru import logger

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.ojin.video import OjinPersonaInitializedFrame
from pipecat.pipeline.task import FrameProcessor
from pipecat.processors.frame_processor import FrameDirection


class MockTTSProcessor(FrameProcessor):
    def __init__(self, config: dict):
        super().__init__()
        self.audio_sequence = config.get(
            "audio_sequence", []
        )  # List of (file_path, start_time) tuples
        self.event_sequence = config.get("event_sequence", [])  # List of (event, start_time) tuples
        self.chunk_size = config.get("chunk_size", 1024)
        self.chunk_delay = config.get("chunk_delay", 0.05)
        self.input_sample_rate = 16000

        logger.debug("Initialized MockTTSProcessor with:")
        logger.debug(f"  Audio sequence: {self.audio_sequence}")
        logger.debug(f"  Chunk size: {self.chunk_size}")
        logger.debug(f"  Chunk delay: {self.chunk_delay}")

        self._running = False
        self._task = None
        self._current_wave_file = None
        self._current_sequence_idx = 0
        self._start_time = 0
        self._last_chunk_time = 0
        self._resampler = create_default_resampler()
        self._persona_initialized = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, EndFrame):
            await self.stop(frame)
            self._running = False

        elif isinstance(frame, StartFrame):
            await self.start(frame)

        elif isinstance(frame, OjinPersonaInitializedFrame):
            logger.debug("MockTTSProcessor received OjinPersonaInitializedFrame, starting timeline")
            self._persona_initialized.set()

        await self.push_frame(frame, direction)

    async def start(self, frame: StartFrame):
        self._running = True
        self._current_sequence_idx = 0
        self._task = self.create_task(self._wait_and_start())
        logger.debug("MockTTSProcessor waiting for OjinPersonaInitializedFrame")

    async def _wait_and_start(self):
        """Wait for persona initialization before starting the timeline."""
        await self._persona_initialized.wait()
        self._start_time = time.monotonic()
        self._last_chunk_time = self._start_time
        logger.debug(f"MockTTSProcessor timeline started at time {self._start_time}")
        self._audio_task = self.create_task(self._simulate_audio())
        self._events_task = self.create_task(self._simulate_events())

    async def stop(self, frame: EndFrame):
        logger.debug("MockTTSProcessor stopping")
        self._running = False
        if self._task:
            await self.cancel_task(self._task)
            self._task = None
        if self._current_wave_file:
            self._current_wave_file.close()
            self._current_wave_file = None

    async def cancel(self, frame: CancelFrame):
        logger.debug("MockTTSProcessor canceling")
        self._running = False
        if self._task:
            await self.cancel_task(self._task)
            self._task = None
        await super().cancel(frame)
        if self._current_wave_file:
            self._current_wave_file.close()
            self._current_wave_file = None
        logger.debug("MockTTSProcessor canceled")

    async def _simulate_events(self):
        """Simulate events based on the event_sequence configuration."""
        if not self.event_sequence:
            return

        current_idx = 0
        while self._running and current_idx < len(self.event_sequence):
            curr_time = time.monotonic()
            elapsed = curr_time - self._start_time

            event, start_time = self.event_sequence[current_idx]

            if elapsed >= start_time:
                logger.debug(f"Triggering event {event} at {elapsed:.2f}s")

                if event == "user_started_speaking":
                    await self.push_frame(StartInterruptionFrame(), FrameDirection.DOWNSTREAM)
                    if self._current_wave_file:
                        self._current_wave_file.close()
                        self._current_wave_file = None
                elif event == "user_stopped_speaking":
                    await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
                else:
                    logger.warning(f"Unknown event type: {type(event)}")

                current_idx += 1
            else:
                await asyncio.sleep(min(0.05, start_time - elapsed))

    async def _simulate_audio(self):
        # await self.push_frame(StartInterruptionFrame(), FrameDirection.DOWNSTREAM)
        # await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        while self._running:
            curr_time = time.monotonic()
            # Respect the chunk delay
            if curr_time - self._last_chunk_time < self.chunk_delay:
                await asyncio.sleep(self.chunk_delay - (curr_time - self._last_chunk_time))
                continue
            self._last_chunk_time = time.monotonic()
            elapsed = self._last_chunk_time - self._start_time
            # Add debug log to show current timing
            # logger.debug(f"Current time: {elapsed:.2f}s, Next file time: {self.audio_sequence[self._current_sequence_idx][1] if self._current_sequence_idx < len(self.audio_sequence) else 'None'}")

            # Check if it's time to switch to the next audio file in the sequence
            while (
                self._current_sequence_idx < len(self.audio_sequence)
                and elapsed >= self.audio_sequence[self._current_sequence_idx][1]
            ):
                if self._current_wave_file:
                    self._current_wave_file.close()
                    self._current_wave_file = None
                file_path = self.audio_sequence[self._current_sequence_idx][0]
                abs_path = os.path.abspath(file_path)
                logger.debug(f"{elapsed} Attempting to open audio file: {abs_path}")
                try:
                    self._current_wave_file = wave.open(file_path, "rb")
                    self._current_wave_file.setpos(0)
                    if self._current_wave_file.getsampwidth() != 2:
                        raise ValueError(
                            f"Unsupported audio format in {file_path}. Expected 16-bit PCM"
                        )
                    logger.debug(f"Successfully opened audio file: {abs_path}")
                    logger.debug(f"  Channels: {self._current_wave_file.getnchannels()}")
                    logger.debug(f"  Sample width: {self._current_wave_file.getsampwidth()}")
                    logger.debug(f"  Frame rate: {self._current_wave_file.getframerate()}")
                    logger.debug(f"  Frames: {self._current_wave_file.getnframes()}")

                    await self.push_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
                except Exception as e:
                    logger.error(f"Error opening audio file {abs_path}: {str(e)}")
                    self._current_wave_file = None
                self._current_sequence_idx += 1

            if self._current_wave_file:
                raw_data = self._current_wave_file.readframes(self.chunk_size)

                frame_obj = TTSAudioRawFrame(
                    audio=raw_data,
                    sample_rate=16000,
                    num_channels=1,
                )

                # logger.info("Pushing audio frame")
                # Use await since we're already in an async context
                await self.push_frame(frame_obj)

                eod_raw_data = self._current_wave_file.readframes(self.chunk_size)
                if len(eod_raw_data) == 0:
                    logger.debug(
                        f"End of file reached for audio file at sequence idx {self._current_sequence_idx - 1}"
                    )
                    self._current_wave_file.close()
                    self._current_wave_file = None
                    await asyncio.sleep(1.5)
                    await self.push_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

                    # Simulate audio coming after TTSStoppedFrame
                    await asyncio.sleep(1.0)

                    # Push silence for 1 second after TTSStoppedFrame to cover race conditions from pipecat TTS Service
                    silence = b"\x00" * 16000
                    frame_obj = TTSAudioRawFrame(
                        audio=silence,
                        sample_rate=16000,
                        num_channels=1,
                    )

                    await self.push_frame(frame_obj)

                    continue
                await asyncio.sleep(self.chunk_delay)
            else:
                if self._current_sequence_idx >= len(self.audio_sequence):
                    logger.debug("No more files in audio sequence, stopping")
                    self._running = False

                await asyncio.sleep(self.chunk_delay)
