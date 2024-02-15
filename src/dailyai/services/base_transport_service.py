from abc import abstractmethod
import asyncio
import itertools
import logging
import queue
import threading
import time
from typing import AsyncGenerator

from dailyai.queue_frame import (
    AudioQueueFrame,
    ChatMessageQueueFrame,
    EndStreamQueueFrame,
    ImageQueueFrame,
    QueueFrame,
    SpriteQueueFrame,
    StartStreamQueueFrame,
)


class BaseTransportService():

    def __init__(
        self,
        **kwargs,
    ) -> None:
        self._mic_enabled = kwargs.get("mic_enabled") or False
        self._mic_sample_rate = kwargs.get("mic_sample_rate") or 16000
        self._camera_enabled = kwargs.get("camera_enabled") or False
        self._camera_width = kwargs.get("camera_width") or 1024
        self._camera_height = kwargs.get("camera_height") or 768
        self._speaker_enabled = kwargs.get("speaker_enabled") or False
        self._speaker_sample_rate = kwargs.get("speaker_sample_rate") or 16000
        self._fps = kwargs.get("fps") or 8

        duration_minutes = kwargs.get("duration_minutes") or 10
        self._expiration = time.time() + duration_minutes * 60

        self.send_queue = asyncio.Queue()
        self.receive_queue = asyncio.Queue()

        self._threadsafe_send_queue = queue.Queue()

        self._images = None

        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        self._stop_threads = threading.Event()
        self._is_interrupted = threading.Event()

        self._logger: logging.Logger = logging.getLogger()

    async def run(self):
        self._prerun()

        async_output_queue_marshal_task = asyncio.create_task(
            self._marshal_frames())

        self._camera_thread = threading.Thread(
            target=self._run_camera, daemon=True)
        self._camera_thread.start()

        self._frame_consumer_thread = threading.Thread(
            target=self._frame_consumer, daemon=True)
        self._frame_consumer_thread.start()

        if self._speaker_enabled:
            self._receive_audio_thread = threading.Thread(
                target=self._receive_audio, daemon=True)
            self._receive_audio_thread.start()

        try:
            while (
                time.time() < self._expiration
                and not self._stop_threads.is_set()
            ):
                await asyncio.sleep(1)
        except Exception as e:
            self._logger.error(f"Exception {e}")
            raise e
        self._stop_threads.set()

        await self.send_queue.put(EndStreamQueueFrame())
        await async_output_queue_marshal_task
        await self.send_queue.join()
        self._frame_consumer_thread.join()

        if self._speaker_enabled:
            self._receive_audio_thread.join()

    def stop(self):
        self._stop_threads.set()

    async def stop_when_done(self):
        await self._wait_for_send_queue_to_empty()
        self.stop()

    async def _wait_for_send_queue_to_empty(self):
        await self.send_queue.join()
        self._threadsafe_send_queue.join()

    @abstractmethod
    def write_frame_to_camera(self, frame: bytes):
        pass

    @abstractmethod
    def write_frame_to_mic(self, frame: bytes):
        pass

    @abstractmethod
    def read_audio_frames(self, desired_frame_count):
        return bytes()

    @abstractmethod
    def _prerun(self):
        pass

    async def _marshal_frames(self):
        while True:
            frame: QueueFrame | list = await self.send_queue.get()
            self._threadsafe_send_queue.put(frame)
            self.send_queue.task_done()
            if isinstance(frame, EndStreamQueueFrame):
                break

    def interrupt(self):
        self._is_interrupted.set()

    async def get_receive_frames(self) -> AsyncGenerator[QueueFrame, None]:
        while True:
            frame = await self.receive_queue.get()
            yield frame
            if isinstance(frame, EndStreamQueueFrame):
                break

    def _receive_audio(self):
        if not self._loop:
            self._logger.error("No loop available for audio thread")
            return

        seconds = 1
        desired_frame_count = self._speaker_sample_rate * seconds
        while not self._stop_threads.is_set():
            buffer = self.read_audio_frames(desired_frame_count)
            if len(buffer) > 0:
                frame = AudioQueueFrame(buffer)
                asyncio.run_coroutine_threadsafe(
                    self.receive_queue.put(frame), self._loop
                )
        asyncio.run_coroutine_threadsafe(
            self.receive_queue.put(EndStreamQueueFrame()), self._loop
        )

    def _set_image(self, image: bytes):
        self._images = itertools.cycle([image])

    def _set_images(self, images: list[bytes], start_frame=0):
        self._images = itertools.cycle(images)

    def _run_camera(self):
        try:
            while not self._stop_threads.is_set():
                if self._images:
                    this_frame = next(self._images)
                    self.write_frame_to_camera(this_frame)

                time.sleep(1.0 / self._fps)
        except Exception as e:
            self._logger.error(f"Exception {e} in camera thread.")
            raise e

    def _frame_consumer(self):
        self._logger.info("🎬 Starting frame consumer thread")
        b = bytearray()
        smallest_write_size = 3200
        all_audio_frames = bytearray()
        while True:
            try:
                frames_or_frame: QueueFrame | list[QueueFrame] = (
                    self._threadsafe_send_queue.get()
                )
                if isinstance(frames_or_frame, QueueFrame):
                    frames: list[QueueFrame] = [frames_or_frame]
                elif isinstance(frames_or_frame, list):
                    frames: list[QueueFrame] = frames_or_frame
                else:
                    raise Exception("Unknown type in output queue")

                for frame in frames:
                    if isinstance(frame, EndStreamQueueFrame):
                        self._logger.info("Stopping frame consumer thread")
                        self._threadsafe_send_queue.task_done()
                        return

                    # if interrupted, we just pull frames off the queue and discard them
                    if not self._is_interrupted.is_set():
                        if frame:
                            if isinstance(frame, AudioQueueFrame):
                                chunk = frame.data

                                all_audio_frames.extend(chunk)

                                b.extend(chunk)
                                truncated_length: int = len(b) - (
                                    len(b) % smallest_write_size
                                )
                                if truncated_length:
                                    self.write_frame_to_mic(
                                        bytes(b[:truncated_length]))
                                    b = b[truncated_length:]
                            elif isinstance(frame, ImageQueueFrame):
                                self._set_image(frame.image)
                            elif isinstance(frame, SpriteQueueFrame):
                                self._set_images(frame.images)
                            elif isinstance(frame, ChatMessageQueueFrame):
                                self._send_chat_message(frame)
                        elif len(b):
                            self.write_frame_to_mic(bytes(b))
                            b = bytearray()
                    else:
                        # if there are leftover audio bytes, write them now; failing to do so
                        # can cause static in the audio stream.
                        if len(b):
                            truncated_length = len(b) - (len(b) % 160)
                            self.write_frame_to_mic(
                                bytes(b[:truncated_length]))
                            b = bytearray()

                        if isinstance(frame, StartStreamQueueFrame):
                            self._is_interrupted.clear()

                self._threadsafe_send_queue.task_done()
            except queue.Empty:
                if len(b):
                    self.write_frame_to_mic(bytes(b))

                b = bytearray()
            except Exception as e:
                print(
                    f"Exception in frame_consumer: {e}, {len(b)}")
                raise e
