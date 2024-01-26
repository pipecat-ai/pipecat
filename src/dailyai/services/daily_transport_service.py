import asyncio
import inspect
import logging
import threading
import time
import types

from functools import partial
from queue import Queue, Empty
from typing import AsyncGenerator

from dailyai.queue_frame import (
    AudioQueueFrame,
    EndStreamQueueFrame,
    ImageQueueFrame,
    QueueFrame,
    StartStreamQueueFrame,
    TranscriptionQueueFrame,
)

from threading import Thread, Event

from daily import (
    EventHandler,
    CallClient,
    Daily,
    VirtualCameraDevice,
    VirtualMicrophoneDevice,
    VirtualSpeakerDevice,
)


class DailyTransportService(EventHandler):
    _daily_initialized = False
    _lock = threading.Lock()

    speaker_enabled: bool
    speaker_sample_rate: int

    def __init__(
        self,
        room_url: str,
        token: str | None,
        bot_name: str,
        duration: float = 10,
        min_others_count: int = 1,
        start_transcription: bool = True,
        speaker_enabled: bool = False,
        speaker_sample_rate: int = 16000,
    ):
        super().__init__()
        self.bot_name: str = bot_name
        self.room_url: str = room_url
        self.token: str | None = token
        self.duration: float = duration
        self.expiration = time.time() + duration * 60
        self.min_others_count = min_others_count
        self.start_transcription = start_transcription

        # This queue is used to marshal frames from the async send queue to the thread that emits audio & video.
        # We need this to maintain the asynchronous behavior of asyncio queues -- to give async functions
        # a chance to run while waiting for queue items -- but also to maintain thread safety and have a threaded
        # handler to send frames, to ensure that sending isn't subject to pauses in the async thread.
        self.threadsafe_send_queue = Queue()

        self.is_interrupted = Event()
        self.stop_threads = Event()
        self.story_started = False
        self.mic_enabled = False
        self.mic_sample_rate = 16000
        self.camera_width = 1024
        self.camera_height = 768
        self.camera_enabled = False
        self.speaker_enabled = speaker_enabled
        self.speaker_sample_rate = speaker_sample_rate

        self.send_queue = asyncio.Queue()
        self.receive_queue = asyncio.Queue()

        self.other_participant_has_joined = False
        self.my_participant_id = None

        self.camera_thread = None
        self.frame_consumer_thread = None

        self.transcription_settings = {
            "language": "en",
            "tier": "nova",
            "model": "2-conversationalai",
            "profanity_filter": True,
            "redact": False,
            "extra": {
                "endpointing": True,
                "punctuate": False,
            },
        }

        self.logger: logging.Logger = logging.getLogger("dailyai")

        self.event_handlers = {}

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    def patch_method(self, event_name, *args, **kwargs):
        try:
            for handler in self.event_handlers[event_name]:
                if inspect.iscoroutinefunction(handler):
                    if self.loop:
                        asyncio.run_coroutine_threadsafe(handler(*args, **kwargs), self.loop)
                    else:
                        raise Exception(
                            "No event loop to run coroutine. In order to use async event handlers, you must run the DailyTransportService in an asyncio event loop.")
                else:
                    handler(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Exception in event handler {event_name}: {e}")
            raise e

    def add_event_handler(self, event_name: str, handler):
        if not event_name.startswith("on_"):
            raise Exception(f"Event handler {event_name} must start with 'on_'")

        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        if event_name not in [method[0] for method in methods]:
            raise Exception(f"Event handler {event_name} not found")

        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = [
                getattr(
                    self, event_name), types.MethodType(
                    handler, self)]
            setattr(self, event_name, partial(self.patch_method, event_name))
        else:
            self.event_handlers[event_name].append(types.MethodType(handler, self))

    def event_handler(self, event_name: str):
        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler

        return decorator

    def configure_daily(self):
        # Only initialize Daily once
        if not DailyTransportService._daily_initialized:
            with DailyTransportService._lock:
                Daily.init()
                DailyTransportService._daily_initialized = True
        self.client = CallClient(event_handler=self)

        if self.mic_enabled:
            self.mic: VirtualMicrophoneDevice = Daily.create_microphone_device(
                "mic", sample_rate=self.mic_sample_rate, channels=1
            )

        if self.camera_enabled:
            self.camera: VirtualCameraDevice = Daily.create_camera_device(
                "camera", width=self.camera_width, height=self.camera_height, color_format="RGB"
            )

        if self.speaker_enabled:
            self.speaker: VirtualSpeakerDevice = Daily.create_speaker_device(
                "speaker", sample_rate=self.speaker_sample_rate, channels=1
            )
            Daily.select_speaker_device("speaker")

        self.image: bytes | None = None
        self.camera_thread = Thread(target=self.run_camera, daemon=True)
        self.camera_thread.start()

        self.logger.info("Starting frame consumer thread")
        self.frame_consumer_thread = Thread(target=self.frame_consumer, daemon=True)
        self.frame_consumer_thread.start()

        self.client.set_user_name(self.bot_name)
        self.client.join(self.room_url, self.token, completion=self.call_joined)
        self.my_participant_id = self.client.participants()["local"]["id"]

        self.client.update_inputs(
            {
                "camera": {
                    "isEnabled": True,
                    "settings": {
                        "deviceId": "camera",
                    },
                },
                "microphone": {
                    "isEnabled": True,
                    "settings": {
                        "deviceId": "mic",
                        "customConstraints": {
                            "autoGainControl": {"exact": False},
                            "echoCancellation": {"exact": False},
                            "noiseSuppression": {"exact": False},
                        },
                    },
                },
            }
        )

        self.client.update_publishing(
            {
                "camera": {
                    "sendSettings": {
                        "maxQuality": "low",
                        "encodings": {
                            "low": {
                                "maxBitrate": 250000,
                                "scaleResolutionDownBy": 1.333,
                                "maxFramerate": 8,
                            }
                        },
                    }
                }
            }
        )

        if self.token and self.start_transcription:
            self.client.start_transcription(self.transcription_settings)

    def _receive_audio(self):
        """Receive audio from the Daily call and put it on the receive queue"""
        seconds = 1
        desired_frame_count = self.speaker_sample_rate * seconds
        while True:
            buffer = self.speaker.read_frames(desired_frame_count)
            if len(buffer) > 0:
                frame = AudioQueueFrame(buffer)
                if self.loop:
                    asyncio.run_coroutine_threadsafe(self.receive_queue.put(frame), self.loop)

    def interrupt(self):
        self.is_interrupted.set()

    async def get_receive_frames(self) -> AsyncGenerator[QueueFrame, None]:
        while True:
            frame = await self.receive_queue.get()
            yield frame
            if isinstance(frame, EndStreamQueueFrame):
                break

    def get_async_send_queue(self):
        return self.send_queue

    async def marshal_frames(self):
        while True:
            frame: QueueFrame | list = await self.send_queue.get()
            self.threadsafe_send_queue.put(frame)
            self.send_queue.task_done()
            if isinstance(frame, EndStreamQueueFrame):
                break

    async def wait_for_send_queue_to_empty(self):
        await self.send_queue.join()
        self.threadsafe_send_queue.join()

    async def stop_when_done(self):
        await self.wait_for_send_queue_to_empty()
        self.stop()

    async def run(self) -> None:
        self.configure_daily()

        self.do_shutdown = False

        async_output_queue_marshal_task = asyncio.create_task(self.marshal_frames())

        try:
            participant_count: int = len(self.client.participants())
            self.logger.info(f"{participant_count} participants in room")
            while time.time() < self.expiration and not self.do_shutdown and not self.stop_threads.is_set():
                await asyncio.sleep(1)
        except Exception as e:
            self.logger.error(f"Exception {e}")
            raise e
        finally:
            self.client.leave()

        self.stop_threads.set()

        await self.receive_queue.put(EndStreamQueueFrame())
        await self.send_queue.put(EndStreamQueueFrame())
        await async_output_queue_marshal_task

        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join()
        if self.frame_consumer_thread and self.frame_consumer_thread.is_alive():
            self.frame_consumer_thread.join()

    def stop(self):
        self.stop_threads.set()

    def on_first_other_participant_joined(self):
        pass

    def call_joined(self, join_data, client_error):
        self.logger.info(f"Call_joined: {join_data}, {client_error}")
        if self.speaker_enabled:
            t = Thread(target=self._receive_audio, daemon=True)
            t.start()

    def on_error(self, error):
        self.logger.error(f"on_error: {error}")

    def on_call_state_updated(self, state):
        pass

    def on_participant_joined(self, participant):
        if not self.other_participant_has_joined and participant["id"] != self.my_participant_id:
            self.other_participant_has_joined = True
            self.on_first_other_participant_joined()

    def on_participant_left(self, participant, reason):
        if len(self.client.participants()) < self.min_others_count + 1:
            self.do_shutdown = True
        pass

    def on_app_message(self, message, sender):
        pass

    def on_transcription_message(self, message: dict):
        if self.loop:
            participantId = ""
            if "participantId" in message:
                participantId = message["participantId"]
            elif "session_id" in message:
                participantId = message["session_id"]
            frame = TranscriptionQueueFrame(message["text"], participantId, message["timestamp"])
            asyncio.run_coroutine_threadsafe(self.receive_queue.put(frame), self.loop)

    def on_transcription_stopped(self, stopped_by, stopped_by_error):
        pass

    def on_transcription_error(self, message):
        pass

    def on_transcription_started(self, status):
        pass

    def set_image(self, image: bytes):
        self.image: bytes | None = image

    def run_camera(self):
        try:
            while not self.stop_threads.is_set():
                if self.image:
                    self.camera.write_frame(self.image)

                time.sleep(1.0 / 8)  # 8 fps
        except Exception as e:
            self.logger.error(f"Exception {e} in camera thread.")
            raise e

    def frame_consumer(self):
        self.logger.info("🎬 Starting frame consumer thread")
        b = bytearray()
        smallest_write_size = 3200
        all_audio_frames = bytearray()
        while True:
            try:
                frames_or_frame: QueueFrame | list[QueueFrame] = self.threadsafe_send_queue.get()
                if isinstance(frames_or_frame, QueueFrame):
                    frames: list[QueueFrame] = [frames_or_frame]
                elif isinstance(frames_or_frame, list):
                    frames: list[QueueFrame] = frames_or_frame
                else:
                    raise Exception("Unknown type in output queue")

                for frame in frames:
                    if isinstance(frame, EndStreamQueueFrame):
                        self.logger.info("Stopping frame consumer thread")
                        self.threadsafe_send_queue.task_done()
                        return

                    # if interrupted, we just pull frames off the queue and discard them
                    if not self.is_interrupted.is_set():
                        if frame:
                            if isinstance(frame, AudioQueueFrame):
                                chunk = frame.data

                                all_audio_frames.extend(chunk)

                                b.extend(chunk)
                                l = len(b) - (len(b) % smallest_write_size)
                                if l:
                                    self.mic.write_frames(bytes(b[:l]))
                                    b = b[l:]
                            elif isinstance(frame, ImageQueueFrame):
                                self.set_image(frame.image)
                        elif len(b):
                            self.mic.write_frames(bytes(b))
                            b = bytearray()
                    else:
                        # if there are leftover audio bytes, write them now; failing to do so
                        # can cause static in the audio stream.
                        if len(b):
                            self.mic.write_frames(bytes(b))
                            b = bytearray()

                        if isinstance(frame, StartStreamQueueFrame):
                            self.is_interrupted.clear()

                self.threadsafe_send_queue.task_done()
            except Empty:
                try:
                    if len(b):
                        self.mic.write_frames(bytes(b))
                except Exception as e:
                    self.logger.error(f"Exception in frame_consumer: {e}, {len(b)}")
                    raise e

                b = bytearray()
            except Exception as e:
                print("!!!!", e)
                raise e
