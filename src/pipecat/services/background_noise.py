import asyncio
import json
import time
from asyncio import sleep
from io import BytesIO

import loguru

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pydub import AudioSegment
from pipecat.frames.frames import AudioRawFrame, OutputAudioRawFrame, Frame, BotStartedSpeakingFrame, \
    BotStoppedSpeakingFrame, EndFrame


class BackgroundNoiseEffect(FrameProcessor):
    def __init__(self, websocket_client, stream_sid, music_path):
        super().__init__(sync=False)
        self._speaking = True
        self._audio_task = self.get_event_loop().create_task(self._audio_task_handler())
        self._audio_queue = asyncio.Queue()
        self._stop = False
        self.stream_sid = stream_sid
        self.websocket_client = websocket_client
        self.music_path = music_path
        self.get_music_part_gen = self._get_music_part()
        self.emptied = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._speaking = True

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._speaking = False
            self.emptied = False

        if isinstance(frame, AudioRawFrame) and self._speaking:
            if not self.emptied:
                self.emptied = True
                buffer_clear_message = {"event": "clear", "streamSid": self.stream_sid}
                await self.websocket_client.send_text(json.dumps(buffer_clear_message))

            frame.audio = self._combine_with_music(frame)

        if isinstance(frame, EndFrame):
            self._stop = True

        await self.push_frame(frame, direction)

    def _combine_with_music(self, frame: AudioRawFrame):
        """
        Combines small raw audio segments from the frame with chunks of a music file.
        """
        small_audio_bytes = frame.audio
        music_audio = AudioSegment.from_wav(self.music_path)
        music_audio = music_audio - 15

        music_position = 0
        small_audio = AudioSegment(
            data=small_audio_bytes,
            sample_width=2,
            frame_rate=16000,
            channels=1
        )

        small_audio_length = len(small_audio)
        music_chunk = music_audio[music_position:music_position + small_audio_length]

        if len(music_chunk) < small_audio_length:
            music_position = 0
            music_chunk += music_audio[:small_audio_length - len(music_chunk)]

        combined_audio = music_chunk.overlay(small_audio)
        music_position += small_audio_length

        output_buffer = BytesIO()
        try:
            combined_audio.export(output_buffer, format="raw")
            return output_buffer.getvalue()
        finally:
            output_buffer.close()

    def _get_music_part(self):
        """
        Generator that yields chunks of background music audio.
        """
        music_audio = AudioSegment.from_wav(self.music_path)
        music_audio = music_audio - 15

        music_position = 0
        small_audio_length = 6400

        while True:
            if music_position + small_audio_length > len(music_audio):
                music_chunk = music_audio[music_position:] + music_audio[
                                                             :(music_position + small_audio_length) % len(music_audio)]
                music_position = (music_position + small_audio_length) % len(music_audio)
            else:
                music_chunk = music_audio[music_position:music_position + small_audio_length]
                music_position += small_audio_length

            output_buffer = BytesIO()
            try:
                music_chunk.export(output_buffer, format="raw")
                frame = OutputAudioRawFrame(audio=output_buffer.getvalue(), sample_rate=16000, num_channels=1)
                yield frame
            finally:
                output_buffer.close()

    async def _audio_task_handler(self):
        while True:
            await sleep(0.005)
            if self._stop:
                break

            if not self._speaking:
                frame = next(self.get_music_part_gen)
                await self.push_frame(frame, FrameDirection.DOWNSTREAM)

