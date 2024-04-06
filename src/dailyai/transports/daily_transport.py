import asyncio
import inspect
import logging
import signal
import threading
import types

from functools import partial
from typing import Any

from dailyai.pipeline.frames import (
    ReceivedAppMessageFrame,
    TranscriptionFrame,
)

from threading import Event

try:
    from daily import (
        EventHandler,
        CallClient,
        Daily,
        VirtualCameraDevice,
        VirtualMicrophoneDevice,
        VirtualSpeakerDevice,
    )
except ModuleNotFoundError as e:
    print(f"Exception: {e}")
    print(
        "In order to use the Daily transport, you need to `pip install dailyai[daily]`.")
    raise Exception(f"Missing module: {e}")


from dailyai.transports.threaded_transport import ThreadedTransport

NUM_CHANNELS = 1

SPEECH_THRESHOLD = 0.90
VAD_RESET_PERIOD_MS = 2000


class DailyTransport(ThreadedTransport, EventHandler):
    _daily_initialized = False
    _lock = threading.Lock()

    _speaker_enabled: bool
    _speaker_sample_rate: int
    _vad_enabled: bool

    # This is necessary to override EventHandler's __new__ method.
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(
        self,
        room_url: str,
        token: str | None,
        bot_name: str,
        min_others_count: int = 1,
        start_transcription: bool = False,
        **kwargs,
    ):
        kwargs['has_webrtc_vad'] = True
        # This will call ThreadedTransport.__init__ method, not EventHandler
        super().__init__(**kwargs)

        self._room_url: str = room_url
        self._bot_name: str = bot_name
        self._token: str | None = token
        self._min_others_count = min_others_count
        self._start_transcription = start_transcription

        self._is_interrupted = Event()
        self._stop_threads = Event()

        self._other_participant_has_joined = False
        self._my_participant_id = None

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

        self._logger: logging.Logger = logging.getLogger("dailyai")

        self._event_handlers = {}

        self.webrtc_vad = Daily.create_native_vad(
            reset_period_ms=VAD_RESET_PERIOD_MS,
            sample_rate=self._speaker_sample_rate,
            channels=NUM_CHANNELS
        )

    def _patch_method(self, event_name, *args, **kwargs):
        try:
            for handler in self._event_handlers[event_name]:
                if inspect.iscoroutinefunction(handler):
                    if self._loop:
                        future = asyncio.run_coroutine_threadsafe(
                            handler(*args, **kwargs), self._loop)

                        # wait for the coroutine to finish. This will also
                        # raise any exceptions raised by the coroutine.
                        future.result()
                    else:
                        raise Exception(
                            "No event loop to run coroutine. In order to use async event handlers, you must run the DailyTransportService in an asyncio event loop.")
                else:
                    handler(*args, **kwargs)
        except Exception as e:
            self._logger.error(f"Exception in event handler {event_name}: {e}")
            raise e

    def _webrtc_vad_analyze(self):
        buffer = self.read_audio_frames(int(self._vad_samples))
        if len(buffer) > 0:
            confidence = self.webrtc_vad.analyze_frames(buffer)
            # yeses = int(confidence * 20.0)
            # nos = 20 - yeses
            # out = "!" * yeses + "." * nos
            # print(f"!!! confidence: {out} {confidence}")
            talking = confidence > SPEECH_THRESHOLD
            return talking

    def add_event_handler(self, event_name: str, handler):
        if not event_name.startswith("on_"):
            raise Exception(
                f"Event handler {event_name} must start with 'on_'")

        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        if event_name not in [method[0] for method in methods]:
            raise Exception(f"Event handler {event_name} not found")

        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = [
                getattr(
                    self, event_name), types.MethodType(
                    handler, self)]
            setattr(self, event_name, partial(self._patch_method, event_name))
        else:
            self._event_handlers[event_name].append(
                types.MethodType(handler, self))

    def event_handler(self, event_name: str):
        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler

        return decorator

    def write_frame_to_camera(self, frame: bytes):
        self.camera.write_frame(frame)

    def write_frame_to_mic(self, frame: bytes):
        self.mic.write_frames(frame)

    def send_app_message(self, message: Any, participantId: str | None):
        self.client.send_app_message(message, participantId)

    def read_audio_frames(self, desired_frame_count):
        bytes = self._speaker.read_frames(desired_frame_count)
        return bytes

    def _prerun(self):
        # Only initialize Daily once
        if not DailyTransport._daily_initialized:
            with DailyTransport._lock:
                Daily.init()
                DailyTransport._daily_initialized = True
        self.client = CallClient(event_handler=self)

        if self._mic_enabled:
            self.mic: VirtualMicrophoneDevice = Daily.create_microphone_device(
                "mic", sample_rate=self._mic_sample_rate, channels=1
            )

        if self._camera_enabled:
            self.camera: VirtualCameraDevice = Daily.create_camera_device(
                "camera", width=self._camera_width, height=self._camera_height, color_format="RGB")

        if self._speaker_enabled or self._vad_enabled:
            self._speaker: VirtualSpeakerDevice = Daily.create_speaker_device(
                "speaker", sample_rate=self._speaker_sample_rate, channels=1
            )
            Daily.select_speaker_device("speaker")

        self.client.set_user_name(self._bot_name)
        self.client.join(
            self._room_url,
            self._token,
            completion=self.call_joined,
            client_settings={
                "inputs": {
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
                },
                "publishing": {
                    "camera": {
                        "sendSettings": {
                            "maxQuality": "low",
                            "encodings": {
                                "low": {
                                    "maxBitrate": self._camera_bitrate,
                                    "scaleResolutionDownBy": 1.333,
                                    "maxFramerate": self._camera_framerate,
                                }
                            },
                        }
                    }
                },
            },
        )
        self._my_participant_id = self.client.participants()["local"]["id"]

        self.client.update_subscription_profiles({
            "base": {
                "camera": "unsubscribed",
            }
        })

        if self._token and self._start_transcription:
            self.client.start_transcription(self.transcription_settings)

        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.process_interrupt_handler)

    def process_interrupt_handler(self, signum, frame):
        self._post_run()
        if callable(self.original_sigint_handler):
            self.original_sigint_handler(signum, frame)

    def _post_run(self):
        self.client.leave()
        self.client.release()

    def on_first_other_participant_joined(self):
        pass

    def call_joined(self, join_data, client_error):
        # self._logger.info(f"Call_joined: {join_data}, {client_error}")
        pass

    def dialout(self, number):
        self.client.start_dialout({"phoneNumber": number})

    def start_recording(self):
        self.client.start_recording()

    def on_error(self, error):
        self._logger.error(f"on_error: {error}")

    def on_call_state_updated(self, state):
        pass

    def on_participant_joined(self, participant):
        if not self._other_participant_has_joined and participant["id"] != self._my_participant_id:
            self._other_participant_has_joined = True
            self.on_first_other_participant_joined()

    def on_participant_left(self, participant, reason):
        if len(self.client.participants()) < self._min_others_count + 1:
            self._stop_threads.set()

    def on_app_message(self, message: Any, sender: str):
        if self._loop:
            frame = ReceivedAppMessageFrame(message, sender)
            print(frame)
            asyncio.run_coroutine_threadsafe(
                self.receive_queue.put(frame), self._loop
            )

    def on_transcription_message(self, message: dict):
        if self._loop:
            participantId = ""
            if "participantId" in message:
                participantId = message["participantId"]
            elif "session_id" in message:
                participantId = message["session_id"]
            if self._my_participant_id and participantId != self._my_participant_id:
                frame = TranscriptionFrame(
                    message["text"], participantId, message["timestamp"])
                asyncio.run_coroutine_threadsafe(
                    self.receive_queue.put(frame), self._loop)

    def on_transcription_error(self, message):
        self._logger.error(f"Transcription error: {message}")

    def on_transcription_started(self, status):
        pass

    def on_transcription_stopped(self, stopped_by, stopped_by_error):
        pass
