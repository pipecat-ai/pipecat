import logging
import os
import time
import wave

from dataclasses import dataclass
from queue import Queue, Empty
from opentelemetry import trace, context

from dailyai.async_processor.async_processor import (
    AsyncProcessor,
    AsyncProcessorState,
    ConversationProcessorCollection,
    Response,
)
from dailyai.services.ai_services import AIServiceConfig
from dailyai.message_handler.message_handler import MessageHandler

from threading import Thread, Semaphore, Event, Timer

from opentelemetry import context
from opentelemetry.context.context import Context

from daily import (
    EventHandler,
    CallClient,
    Daily,
    VirtualCameraDevice,
    VirtualMicrophoneDevice,
    VirtualSpeakerDevice,
)


@dataclass
class OrchestratorConfig:
    room_url: str
    token: str
    bot_name: str
    expiration: float


class Orchestrator(EventHandler):
    def __init__(
        self,
        daily_config: OrchestratorConfig,
        ai_service_config: AIServiceConfig,
        conversation_processors: ConversationProcessorCollection,
        message_handler: MessageHandler,
        tracer=None,
    ):
        self.bot_name: str = daily_config.bot_name
        self.room_url: str = daily_config.room_url
        self.token: str = daily_config.token
        self.expiration: float = daily_config.expiration

        self.logger: logging.Logger = logging.getLogger("bot-instance")
        self.tracer = tracer or trace.get_tracer("orchestrator")

        self.ctx: Context = context.get_current()

        self.transcription = ""
        self.last_fragment_at = None
        self.talked_at = None
        self.paused_at = None

        self.logger.info(f"Creating Response for introductions")
        self.services: AIServiceConfig = ai_service_config
        self.output_queue = Queue()
        self.is_interrupted = Event()
        self.stop_threads = Event()
        self.story_started = False

        self.message_handler = message_handler
        self.conversation_processors: ConversationProcessorCollection = conversation_processors

        if conversation_processors.introduction is not None:
            intro = conversation_processors.introduction(
                services=self.services, message_handler=self.message_handler, output_queue=self.output_queue
            )
            intro.prepare()
            intro.set_state_callback(AsyncProcessorState.DONE, self.on_intro_played)
            intro.set_state_callback(AsyncProcessorState.FINALIZED, self.on_intro_finished)
            self.logger.info(f"Introduction is preparing")

            self.current_response: AsyncProcessor = intro
        self.can_interrupt = False
        # self.response_event.set()
        self.response_semaphore = Semaphore()

        self.speech_timeout = None
        self.interrupt_time = None

        self.logger.info("Configuring daily")
        self.configure_daily()

    def configure_daily(self):
        Daily.init()
        self.client = CallClient(event_handler=self)

        self.logger.info(f"Mic sample rate: {self.services.tts.get_mic_sample_rate()}")
        self.mic: VirtualMicrophoneDevice  = Daily.create_microphone_device(
            "mic", sample_rate=self.services.tts.get_mic_sample_rate(), channels=1
        )
        self.speaker: VirtualSpeakerDevice = Daily.create_speaker_device(
            "speaker", sample_rate=16000, channels=1
        )
        self.camera: VirtualCameraDevice = Daily.create_camera_device(
            "camera", width=720, height=1280, color_format="RGB"
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

    def start(self) -> None:
        # TODO: this loop could, I think, be replaced with a timer and an event
        self.participant_left = False

        try:
            participant_count: int = len(self.client.participants())
            self.logger.info(f"{participant_count} participants in room")
            while time.time() < self.expiration and not self.participant_left:
                # all handling of incoming transcriptions happens in on_transcription_message
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Exception {e}")
        finally:
            self.client.leave()

    def stop(self):
        self.logger.info("Stop current response")
        if self.current_response:
            if self.current_response.state < AsyncProcessorState.INTERRUPTED:
                self.current_response.interrupt()

            self.logger.info("Wait for state transition")
            self.current_response.wait_for_state_transition(AsyncProcessorState.FINALIZED)

        self.stop_threads.set()
        self.camera_thread.join()
        self.logger.info("Camera thread stopped")

        self.logger.info("Put stop in output queue")
        self.output_queue.put({"type": "stop"})

        self.frame_consumer_thread.join()
        self.logger.info("Orchestrator stopped.")

    def on_intro_played(self, intro):
        self.can_interrupt = True
        intro.finalize()

    def on_intro_finished(self, intro):
        pass

    def on_response_played(self, response):
        response.finalize()
        self.display_waiting()

    def on_response_finished(self, response):
        if not response.was_interrupted:
            self.message_handler.finalize_user_message()

    def call_joined(self, join_data, client_error):
        self.logger.info(f"Call_joined: {join_data}, {client_error}")
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
                }
            }
        )

    def on_participant_joined(self, participant):
        with self.tracer.start_as_current_span("on_participant_joined", context=self.ctx):
            self.logger.info(f"on_participant_joined: {participant}")

            # TODO: figure out the architecture to get the story id to the client
            # self.client.send_app_message({"event": "story-id", "storyID": self.story_id})
            time.sleep(2)

            if not self.story_started:
                self.action()
                self.story_started = True

    def on_participant_left(self, participant, reason):
        if len(self.client.participants()) < 2:
            self.logger.info(f"Participant {participant} left")
            self.participant_left = True

    def on_app_message(self, message, sender):
        with self.tracer.start_as_current_span("on_app_message", context=self.ctx):
            self.logger.info(f"on_app_message {message} from {sender}")
            if "isSpeaking" in message and message["isSpeaking"] == True:
                self.handle_user_started_talking()

            if "isSpeaking" in message and message["isSpeaking"] == False:
                self.handle_user_stopped_talking()

    def on_transcription_message(self, message):
        with self.tracer.start_as_current_span("on_transcription_message", context=self.ctx):
            if message["session_id"] != self.my_participant_id:
                self.handle_transcription_fragment(message['text'])

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

                time.sleep(1.0 / 8.0)  # 8 fps
        except Exception as e:
            self.logger.error(f"Exception {e} in camera thread.")

        print("==== camera thread exitings")

    def handle_user_started_talking(self):
        # TODO: allow configuration of the timer timeout
        self.logger.error("user started talking")
        self.speech_timeout = Timer(1.0, self.utterance_interrupt)

    def handle_user_stopped_talking(self):
        self.logger.error("user stopped talking, canceling utterance interrupt")
        if self.speech_timeout:
            self.speech_timeout.cancel()

    def utterance_interrupt(self):
        self.logger.error("utterance interrupt")
        self.is_interrupted.set()

    def handle_transcription_fragment(self, fragment):
        if not self.can_interrupt:
            return

        # start generating a new response. We'll do the fast parts of the interrupt
        # now but wait for the state transition after we've kicked off the prepare
        # on the new response.
        if (
            self.current_response
            and self.current_response.state < AsyncProcessorState.INTERRUPTED
        ):
            self.interrupt_time = time.perf_counter()
            self.is_interrupted.set()
            self.current_response.interrupt()

        self.display_thinking()
        self.message_handler.add_user_message(fragment)

        response_type = self.conversation_processors.response or Response
        new_response: Response = response_type(
            self.services, self.message_handler, self.output_queue
        )
        new_response.set_state_callback(
            AsyncProcessorState.DONE, self.on_response_played
        )
        new_response.set_state_callback(
            AsyncProcessorState.FINALIZED, self.on_response_finished
        )
        new_response.prepare()

        self.response_semaphore.acquire()
        if (
            self.current_response
            and self.current_response.state < AsyncProcessorState.INTERRUPTED
        ):
            self.current_response.wait_for_state_transition(
                AsyncProcessorState.FINALIZED
            )

        self.current_response = new_response
        self.current_response.play()

        self.response_semaphore.release()

    def display_waiting(self):
        # I don't love this design, need to think more about how to do this well
        listening_images = [
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-2",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-2",
            "sc-listen-1",
            "sc-listen-2",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-2",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-2",
            "sc-listen-1",
            "sc-listen-1",
            "sc-listen-1",
        ]
        # self.display_images(listening_images)

    def display_thinking(self):
        thinking_images = [
            "sc-think-1",
            "sc-think-1",
            "sc-think-2",
            "sc-think-2",
            "sc-think-3",
            "sc-think-3",
            "sc-think-4",
            "sc-think-4",
        ]
        # self.display_images(thinking_images)

    def action(self):
        self.logger.info("Starting camera thread")
        self.image: bytes | None = None
        self.camera_thread = Thread(target=self.run_camera, daemon=True)
        self.camera_thread.start()

        self.logger.info("Starting frame consumer thread")
        self.frame_consumer_thread = Thread(target=self.frame_consumer, daemon=True)
        self.frame_consumer_thread.start()

        self.logger.info("Playing introduction")
        self.can_interrupt = False
        self.current_response.play()

    def frame_consumer(self):
        self.logger.info("🎬 Starting frame consumer thread")
        b = bytearray()
        smallest_write_size = 3200
        expected_idx = 0
        all_audio_frames = bytearray()
        while True:
            try:
                frame = self.output_queue.get()
                if frame["type"] == "stop":
                    self.logger.info("Stopping frame consumer thread")

                    if os.getenv("WRITE_BOT_AUDIO", False):
                        filename = f"conversation-{len(all_audio_frames)}.wav"
                        with wave.open(filename, "wb") as f:
                            f.setnchannels(1)
                            f.setframerate(16000)
                            f.setsampwidth(2)
                            f.setcomptype("NONE", "not compressed")
                            f.writeframes(all_audio_frames)
                    return

                if frame["idx"] != expected_idx and frame["idx"] != 0:
                    self.logger.error(
                        f"🎬 Expected frame {expected_idx}, got {frame['idx']}"
                    )

                expected_idx += 1

                # if interrupted, we just pull frames off the queue and discard them
                if not self.is_interrupted.is_set():
                    if frame:
                        if frame["type"] == "audio_frame":
                            chunk = frame["data"]

                            all_audio_frames.extend(chunk)

                            b.extend(chunk)
                            l = len(b) - (len(b) % smallest_write_size)
                            if l:
                                self.mic.write_frames(bytes(b[:l]))
                                b = b[l:]
                        elif frame["type"] == "image_frame":
                            self.set_image(frame["data"])
                    elif len(b):
                        self.mic.write_frames(bytes(b))
                        b = bytearray()
                else:
                    if self.interrupt_time:
                        self.logger.info(f"Lag to stop stream after interruption {time.perf_counter() - self.interrupt_time}")
                        self.interrupt_time = None

                    if frame["type"] == "start_stream":
                        self.is_interrupted.clear()

                self.output_queue.task_done()
            except Empty:
                try:
                    if len(b):
                        self.mic.write_frames(bytes(b))
                except Exception as e:
                    self.logger.error(f"Exception in frame_consumer: {e}, {len(b)}")

                b = bytearray()
