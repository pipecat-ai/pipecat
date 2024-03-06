import array
import io
import math
import time
from typing import AsyncGenerator
import wave
from dailyai.pipeline.frames import AudioFrame, Frame, TranscriptionQueueFrame
from dailyai.services.ai_services import STTService


class LocalSTTService(STTService):
    _content: io.BufferedRandom
    _wave: wave.Wave_write
    _current_silence_frames: int

    # Configuration
    _min_rms: int
    _max_silence_frames: int
    _frame_rate: int

    def __init__(self,
                 min_rms: int = 400,
                 max_silence_frames: int = 3,
                 frame_rate: int = 16000,
                 **kwargs):
        super().__init__(frame_rate, **kwargs)
        self._current_silence_frames = 0
        self._min_rms = min_rms
        self._max_silence_frames = max_silence_frames
        self._frame_rate = frame_rate
        self._new_wave()

    def _new_wave(self):
        """Creates a new wave object and content buffer."""
        self._content = io.BufferedRandom(io.BytesIO())
        ww = wave.open(self._content, "wb")
        ww.setnchannels(1)
        ww.setsampwidth(2)
        ww.setframerate(self._frame_rate)
        self._wave = ww

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Processes a frame of audio data, either buffering or transcribing it."""
        if not isinstance(frame, AudioFrame):
            return

        data = frame.data
        # Try to filter out empty background noise
        # (Very rudimentary approach, can be improved)
        rms = self._get_volume(data)
        if rms >= self._min_rms:
            # If volume is high enough, write new data to wave file
            self._wave.writeframesraw(data)

        # If buffer is not empty and we detect a 3-frame pause in speech,
        # transcribe the audio gathered so far.
        if self._content.tell() > 0 and self._current_silence_frames > self._max_silence_frames:
            self._current_silence_frames = 0
            self._wave.close()
            self._content.seek(0)
            text = await self.run_stt(self._content)
            self._new_wave()
            yield TranscriptionQueueFrame(text, '', str(time.time()))
        # If we get this far, this is a frame of silence
        self._current_silence_frames += 1

    def _get_volume(self, audio: bytes) -> float:
        # https://docs.python.org/3/library/array.html
        audio_array = array.array('h', audio)
        squares = [sample**2 for sample in audio_array]
        mean = sum(squares) / len(audio_array)
        rms = math.sqrt(mean)
        return rms
