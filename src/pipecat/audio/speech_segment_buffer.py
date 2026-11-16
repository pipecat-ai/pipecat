#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A buffer that keeps one stretch of speech, with the audio just before it."""


class SpeechSegmentBuffer:
    """Keeps the audio of one stretch of speech, with the audio just before it.

    Between stretches of speech it keeps only the last ``pre_speech_secs`` of
    audio, which holds the start of the speech the VAD reports late. From
    :meth:`start` to :meth:`stop` it keeps everything, or the last
    ``max_secs`` of it, and :meth:`stop` returns the segment. The audio is
    16-bit mono PCM.

    Example::

        buffer = SpeechSegmentBuffer()
        buffer.set_sample_rate(16000)

        buffer.append(audio)  # every input audio chunk
        buffer.start()  # when the VAD reports speech
        segment = buffer.stop()  # when the VAD reports silence
    """

    def __init__(self, *, pre_speech_secs: float = 1.0, max_secs: float | None = None):
        """Initialize the buffer.

        Args:
            pre_speech_secs: Seconds of audio kept from before speech starts.
                The VAD can report speech most of a second after it starts.
            max_secs: The most seconds a segment keeps, its last ones, or
                None to keep it all.
        """
        self._pre_speech_secs = pre_speech_secs
        self._max_secs = max_secs
        self._pre_speech_size = 0
        self._max_size: int | None = None
        self._buffer = bytearray()
        self._speaking = False

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate of the audio to come.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        self._pre_speech_size = int(sample_rate * self._pre_speech_secs) * 2
        self._max_size = (
            int(sample_rate * self._max_secs) * 2 if self._max_secs is not None else None
        )

    def append(self, audio: bytes):
        """Add audio, dropping the oldest beyond what the buffer keeps.

        Args:
            audio: 16-bit mono PCM audio.
        """
        self._buffer += audio
        limit = self._max_size if self._speaking else self._pre_speech_size
        if limit is not None and len(self._buffer) > limit:
            del self._buffer[: len(self._buffer) - limit]

    def start(self):
        """Start a segment, with the audio kept from before it."""
        self._speaking = True

    def stop(self) -> bytes:
        """End the segment and return its audio, emptying the buffer.

        Returns:
            The segment's 16-bit mono PCM audio.
        """
        self._speaking = False
        segment = bytes(self._buffer)
        self._buffer.clear()
        return segment
