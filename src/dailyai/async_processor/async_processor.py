import json
import logging
import re

from collections import defaultdict
from dataclasses import dataclass, field
from queue import Queue, PriorityQueue, Empty
from threading import Event, Semaphore, Thread
from typing import Any, Generator, Iterator, Optional, Type, TypedDict

from dailyai.services.ai_services import AIServiceConfig
from dailyai.message_handler.message_handler import MessageHandler

frame_idx = 0

class AsyncProcessorState:
    # Setting class variables, other synchronous activities
    INIT = 0

    # Making asynchronous requests to LLM and other services to render response
    PREPARING = 1

    # Ready to start presenting to user (but may not have all data yet)
    READY = 2

    # Playing response
    PLAYING = 3

    # An interrupt has been requested and the response is shutting down in-flight processing
    INTERRUPTING = 4

    # An interrupt has been requested and the response is finished stopping in-flight processing
    INTERRUPTED = 5

    # Response has been played or interrupted
    DONE = 6

    # Response is being finalized (updating records of speech, updating LLM context, etc.)
    FINALIZING = 7

    # Response is complete. This could mean that everything is updated, or that the response
    # was interrupted.
    FINALIZED = 8

    state_transitions = {
        INIT: [PREPARING, INTERRUPTING],
        PREPARING: [READY, INTERRUPTING],
        READY: [PLAYING, INTERRUPTING],
        PLAYING: [DONE, INTERRUPTING],
        INTERRUPTING: [INTERRUPTED],
        INTERRUPTED: [DONE],
        DONE: [FINALIZING],
        FINALIZING: [FINALIZED],
        FINALIZED: [FINALIZED],
    }


@dataclass(order=True)
class StateTransitionItem:
    state: int
    evt: Event = field(compare=False)

class AsyncProcessor:
    def __init__(
        self,
        services: AIServiceConfig
    ) -> None:
        self.state = AsyncProcessorState.INIT
        self.prepare_thread = None
        self.play_thread = None
        self.finalize_thread = None

        self.services: AIServiceConfig = services

        self.state_transition_semaphore = Semaphore()
        self.waiting_for_state_changes = PriorityQueue()
        self.state_queue = Queue()

        self.state_change_callbacks = defaultdict(list)

        self.was_interrupted = False

        self.logger: logging.Logger = logging.getLogger("bot-instance")

    def set_state(self, state: int) -> None:
        if state in AsyncProcessorState.state_transitions[self.state]:
            self.state_transition_semaphore.acquire()

            self.state: int = state
            self.state_transition_semaphore.release()

            # wake up any threads waiting for this state transition
            try:
                while True:
                    waiter = self.waiting_for_state_changes.get_nowait()
                    if waiter.state <= state:
                        waiter.evt.set()
                    else:
                        self.waiting_for_state_changes.put(waiter)
                        break
            except Empty:
                pass

            # make all the callbacks for this state
            for callback in self.state_change_callbacks[state]:
                callback(self)
        else:
            self.logger.error(
                f"Invalid state transition from {self.state} to {state} in {self.__class__.__name__}"
            )
            raise Exception(f"Invalid state transition from {self.state} to {state}")

    #
    # This is used for state transitions that could be blocked by an interruption.
    # If we are interrupted, we silently fail this call. Use only if you know that
    # this state transition should fail if the processor has been interrupted.
    #

    def maybe_set_state(self, state: int) -> bool:
        if state in AsyncProcessorState.state_transitions[self.state]:
            self.set_state(state)
            return True
        else:
            return False

    def wait_for_state_transition(self, state: int) -> None:
        if self.state >= state:
            return

        self.state_transition_semaphore.acquire()

        evt = Event()
        self.waiting_for_state_changes.put(StateTransitionItem(state, evt))
        self.state_transition_semaphore.release()
        result = evt.wait(120.0)
        if not result:
            self.logger.error(
                f"Timed out waiting for state transition to {state} from {self.state}"
            )

    def set_state_callback(self, state: int, callback: callable) -> None:
        self.state_change_callbacks[state].append(callback)

    def prepare(self) -> None:
        self.prepare_thread = Thread(target=self.async_prepare, daemon=True)
        self.prepare_thread.start()
        self.wait_for_state_transition(AsyncProcessorState.READY)

    def play(self) -> None:
        self.wait_for_state_transition(AsyncProcessorState.READY)
        self.play_thread = Thread(target=self.async_play, daemon=True)
        self.play_thread.start()
        self.wait_for_state_transition(AsyncProcessorState.PLAYING)

    def finalize(self) -> None:
        # don't finalize until we're done playing.
        self.wait_for_state_transition(AsyncProcessorState.DONE)
        self.set_state(AsyncProcessorState.FINALIZING)
        self.do_finalization()
        self.set_state(AsyncProcessorState.FINALIZED)

    def interrupt(self) -> None:
        # nothing to interrupt if we're already finalizing or finalized, no-op
        if self.state in [
            AsyncProcessorState.FINALIZING,
            AsyncProcessorState.FINALIZED,
        ]:
            return

        self.set_state(AsyncProcessorState.INTERRUPTING)
        self.was_interrupted = True
        self.do_interruption()
        self.set_state(AsyncProcessorState.INTERRUPTED)
        self.set_state(AsyncProcessorState.DONE)

    def async_play(self) -> None:
        self.logger.info(f"Starting to play")
        if self.maybe_set_state(AsyncProcessorState.PLAYING):
            self.do_play()
        self.maybe_set_state(AsyncProcessorState.DONE)

    def async_prepare(self) -> None:
        self.set_state(AsyncProcessorState.PREPARING)
        self.preparation_iterator = self.get_preparation_iterator()
        self.set_state(AsyncProcessorState.READY)
        for chunk in self.preparation_iterator:
            if self.state not in [
                AsyncProcessorState.READY,
                AsyncProcessorState.PLAYING,
            ]:
                break

            self.process_chunk(chunk)

        self.logger.info(f"Preparation done for {self.__class__.__name__}")
        self.preparation_done()

    def preparation_done(self):
        pass

    def get_preparation_iterator(self) -> Iterator:
        yield None

    def process_chunk(self, chunk) -> None:
        pass

    def do_interruption(self) -> None:
        pass

    def do_play(self) -> None:
        pass

    def do_finalization(self) -> None:
        pass


class Response(AsyncProcessor):
    def __init__(
        self,
        services,
        message_handler,
        output_queue,
    ) -> None:
        super().__init__(services)

        self.message_handler: MessageHandler = message_handler
        self.output_queue: Queue = output_queue
        self.has_sent_first_frame = False

        self.chunks_in_preparation = Queue()

        self.llm_responses: list[str] = []

    def get_preparation_iterator(self) -> Iterator:
        messages_for_llm = self.message_handler.get_llm_messages()
        self.logger.debug(f"Messages for llm: {json.dumps(messages_for_llm, indent=2)}")
        return self.clauses_from_chunks(
            self.services.llm.run_llm_async(messages_for_llm)
        )

    def clauses_from_chunks(self, chunks) -> Iterator:
        out = ""
        for chunk in chunks:
            if self.state not in [
                AsyncProcessorState.READY,
                AsyncProcessorState.PLAYING,
            ]:
                break

            out += chunk

            if re.match(r"^.*[.!?]$", out):  # it looks like a sentence
                yield out.strip()
                out = ""

        if out.strip():
            yield out.strip()

    def get_frames_from_tts_response(self, audio_frame) -> list[dict[str, Any]]:
        return [{"type": "audio_frame", "data": audio_frame}]

    def get_frames_from_chunk(self, chunk) -> Generator[list[dict[str, Any]], Any, None]:
        for audio_frame in self.services.tts.run_tts(chunk):
            yield self.get_frames_from_tts_response(audio_frame)

    def process_chunk(self, chunk) -> None:
        self.chunks_in_preparation.put((chunk, self.get_frames_from_chunk(chunk)))

    def preparation_done(self):
        self.chunks_in_preparation.put((None, None))

    def do_play(self) -> None:
        while True:
            if self.state not in [
                AsyncProcessorState.READY,
                AsyncProcessorState.PLAYING,
            ]:
                break
            prepared_chunk = self.chunks_in_preparation.get()
            if prepared_chunk[0] is None:
                return

            self.play_prepared_chunk(prepared_chunk)

    def play_prepared_chunk(self, prepared_chunk) -> None:
        chunk, tts_generator = prepared_chunk
        global frame_idx
        for frames in tts_generator:
            if self.state not in [
                AsyncProcessorState.READY,
                AsyncProcessorState.PLAYING,
            ]:
                break

            if not self.has_sent_first_frame:
                self.output_queue.put({"type": "start_stream", "idx": frame_idx})
                frame_idx += 1
                self.has_sent_first_frame = True

            for frame in frames:
                frame["idx"] = frame_idx
                self.output_queue.put(frame)
                frame_idx += 1

        self.output_queue.join()
        self.llm_responses.append(chunk)

    def do_finalization(self) -> None:
        self.message_handler.add_assistant_messages(self.llm_responses)

    def do_interruption(self) -> None:
        self.chunks_in_preparation.put((None, None))

        if self.prepare_thread and self.prepare_thread.is_alive():
            self.prepare_thread.join()

        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join()


@dataclass
class ConversationProcessorCollection:
    introduction: Optional[Type[Response]] = None
    waiting: Optional[Type[Response]] = None
    response: Optional[Type[Response]] = None
    goodbye: Optional[Type[Response]] = None
