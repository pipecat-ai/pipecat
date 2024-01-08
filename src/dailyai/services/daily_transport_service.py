import asyncio
import inspect
import logging
import time
import types

from functools import partial
from queue import Queue, Empty

from dailyai.output_queue import OutputQueueFrame, FrameType

from threading import Thread, Event, Timer

from daily import (
    EventHandler,
    CallClient,
    Daily,
    VirtualCameraDevice,
    VirtualMicrophoneDevice,
    VirtualSpeakerDevice,
)

class DailyTransportService(EventHandler):
    def __init__(
        self,
        room_url: str,
        token: str | None,
        bot_name: str,
        duration: float = 10,
    ):
        super().__init__()
        self.bot_name: str = bot_name
        self.room_url: str = room_url
        self.token: str | None = token
        self.duration: float = duration
        self.expiration = time.time() + duration * 60

        self.output_queue = Queue()
        self.is_interrupted = Event()
        self.stop_threads = Event()
        self.story_started = False
        self.mic_enabled = False
        self.mic_sample_rate = 16000
        self.camera_enabled = False

        self.camera_thread = None
        self.frame_consumer_thread = None

        self.logger: logging.Logger = logging.getLogger("dailyai")

        self.event_handlers = {}

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    def patch_method(self, event_name, *args):
        for handler in self.event_handlers[event_name]:
            if inspect.iscoroutinefunction(handler):
                if self.loop:
                    future = asyncio.run_coroutine_threadsafe(handler(*args), self.loop)
                    #concurrent.futures.wait(future)
                else:
                    raise Exception("No event loop to run coroutine. In order to use async event handlers, you must run the DailyTransportService in an asyncio event loop.")
                asyncio.run(handler(*args))
            else:
                handler(*args)

    def add_event_handler(self, event_name: str, handler):
        if not event_name.startswith("on_"):
            raise Exception(f"Event handler {event_name} must start with 'on_'")

        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        if event_name not in [method[0] for method in methods]:
            raise Exception(f"Event handler {event_name} not found")

        if not event_name in self.event_handlers:
            self.event_handlers[event_name] = [getattr(self, event_name), types.MethodType(handler, self)]
            setattr(self, event_name, partial(self.patch_method, event_name))
        else:
            self.event_handlers[event_name].append(types.MethodType(handler, self))

    def event_handler(self, event_name: str):
        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler

        return decorator

    def configure_daily(self):
        Daily.init()
        self.client = CallClient(event_handler=self)

        if self.mic_enabled:
            self.mic: VirtualMicrophoneDevice = Daily.create_microphone_device(
                "mic", sample_rate=self.mic_sample_rate, channels=1
            )

        if self.camera_enabled:
            self.camera: VirtualCameraDevice = Daily.create_camera_device(
                "camera", width=self.camera_width, height=self.camera_height, color_format="RGB"
            )

        self.speaker: VirtualSpeakerDevice = Daily.create_speaker_device(
            "speaker", sample_rate=16000, channels=1
        )

        Daily.select_speaker_device("speaker")

        self.client.set_user_name(self.bot_name)
        self.client.join(self.room_url, self.token, completion=self.call_joined)

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

        self.my_participant_id = self.client.participants()["local"]["id"]

    def run(self) -> None:
        self.configure_daily()
        self.running_thread = Thread(target=self.run_daily, daemon=True)
        self.running_thread.start()
        self.running_thread.join()

    def run_daily(self):
        # TODO: this loop could, I think, be replaced with a timer and an event
        self.participant_left = False

        try:
            participant_count: int = len(self.client.participants())
            self.logger.info(f"{participant_count} participants in room")
            while time.time() < self.expiration and not self.participant_left and not self.stop_threads.is_set():
                # all handling of incoming transcriptions happens in on_transcription_message
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Exception {e}")
        finally:
            self.client.leave()

        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join()
        if self.frame_consumer_thread and self.frame_consumer_thread.is_alive():
            self.output_queue.put(OutputQueueFrame(FrameType.END_STREAM, None))
            self.frame_consumer_thread.join()

    def stop(self):
        self.stop_threads.set()

    def call_joined(self, join_data, client_error):
        self.logger.info(f"Call_joined: {join_data}, {client_error}")

        self.image: bytes | None = None
        self.camera_thread = Thread(target=self.run_camera, daemon=True)
        self.camera_thread.start()

        self.logger.info("Starting frame consumer thread")
        self.frame_consumer_thread = Thread(target=self.frame_consumer, daemon=True)
        self.frame_consumer_thread.start()

        if self.token:
            self.client.start_transcription(
                {
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
            )

    def on_call_state_updated(self, state):
        pass

    def on_participant_joined(self, participant):
        pass

    def on_participant_left(self, participant, reason):
        pass

    def on_app_message(self, message, sender):
        pass

    def on_transcription_message(self, message):
        pass

    def on_transcription_stopped(self, stopped_by, stopped_by_error):
        self.logger.info(f"Transcription stopped {stopped_by}, {stopped_by_error}")

    def on_transcription_error(self, message):
        self.logger.error(f"Transcription error {message}")

    def on_transcription_started(self, status):
        self.logger.info(f"Transcription started {status}")

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

    def frame_consumer(self):
        self.logger.info("🎬 Starting frame consumer thread")
        b = bytearray()
        smallest_write_size = 3200
        all_audio_frames = bytearray()
        while True:
            try:
                frames_or_frame: OutputQueueFrame | list[OutputQueueFrame] = self.output_queue.get()
                if type(frames_or_frame) == OutputQueueFrame:
                    frames: list[OutputQueueFrame] = [frames_or_frame]
                elif type(frames_or_frame) == list:
                    frames: list[OutputQueueFrame] = frames_or_frame
                else:
                    raise Exception("Unknown type in output queue")

                for frame in frames:
                    if frame.frame_type == FrameType.END_STREAM:
                        self.logger.info("Stopping frame consumer thread")
                        return

                    # if interrupted, we just pull frames off the queue and discard them
                    if not self.is_interrupted.is_set():
                        if frame:
                            if frame.frame_type == FrameType.AUDIO_FRAME:
                                chunk = frame.frame_data

                                all_audio_frames.extend(chunk)

                                b.extend(chunk)
                                l = len(b) - (len(b) % smallest_write_size)
                                if l:
                                    self.mic.write_frames(bytes(b[:l]))
                                    b = b[l:]
                            elif frame.frame_type == FrameType.IMAGE_FRAME:
                                self.set_image(frame.frame_data)
                        elif len(b):
                            self.mic.write_frames(bytes(b))
                            b = bytearray()
                    else:
                        if self.interrupt_time:
                            self.logger.info(
                                f"Lag to stop stream after interruption {time.perf_counter() - self.interrupt_time}"
                            )
                            self.interrupt_time = None

                        if frame.frame_type == FrameType.START_STREAM:
                            self.is_interrupted.clear()

                self.output_queue.task_done()
            except Empty:
                try:
                    if len(b):
                        self.mic.write_frames(bytes(b))
                except Exception as e:
                    self.logger.error(f"Exception in frame_consumer: {e}, {len(b)}")

                b = bytearray()
