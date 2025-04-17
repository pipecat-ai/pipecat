#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import itertools
import sys
import time
import numpy as np
from typing import AsyncGenerator, List, Optional, Tuple

from loguru import logger
from PIL import Image

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    MixerControlFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    SpriteFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.time import nanoseconds_to_seconds

BOT_VAD_STOP_SECS = 0.35
_FADE_SIGNAL = object() # Signal object for queue communication

class BaseOutputTransport(FrameProcessor):
    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(**kwargs)
        self._params = params
        self._sink_task: Optional[asyncio.Task] = None
        self._sink_clock_task: Optional[asyncio.Task] = None
        self._camera_out_task: Optional[asyncio.Task] = None
        self._camera_images: Optional[itertools.cycle] = None
        self._sample_rate: int = 0
        self._resampler = create_default_resampler()
        self._audio_chunk_size: int = 0
        # Buffer for audio received via _handle_audio BEFORE it's chunked and put on _sink_queue
        self._audio_buffer = bytearray()
        self._stopped_event = asyncio.Event()
        self._bot_speaking: bool = False

        # --- Tapering State ---
        self._taper_duration_s: float = 5.0 # Default 5 seconds, updated in start()
        self._should_start_fading: bool = False # Signal FROM process_frame TO sink_task check
        self._is_fading: bool = False           # Active fade state IN sink_task
        self._fade_start_time: float = 0.0
        # --- End Tapering State ---

        self._sink_queue: Optional[asyncio.Queue] = None
        self._sink_clock_queue: Optional[asyncio.PriorityQueue] = None
        self._camera_out_queue: Optional[asyncio.Queue] = None
        # Lock to prevent race conditions during task resets after interruption
        self._reset_lock = asyncio.Lock()


    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def start(self, frame: StartFrame):
        self._sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate
        if self._params.audio_out_channels <= 0:
            logger.error("Audio out channels must be positive.")
            self._params.audio_out_channels = 1 # Default
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks
        if self._audio_chunk_size <= 0:
             logger.warning(f"Calculated audio chunk size is {self._audio_chunk_size}. Setting default.")
             self._audio_chunk_size = audio_bytes_10ms * 10 if audio_bytes_10ms > 0 else 1600 # e.g. 100ms at 8kHz mono

        taper_ms = getattr(self._params, 'interrupt_taper_duration_ms', 5000)
        self._taper_duration_s = max(0.01, taper_ms / 1000.0)
        logger.debug(f"Audio taper duration set to: {self._taper_duration_s:.3f} seconds ({taper_ms}ms)")

        if self._params.audio_out_mixer:
            await self._params.audio_out_mixer.start(self._sample_rate)

        # Initial task creation clears queues
        self._create_sink_tasks(preserve_queue=False)
        self._create_camera_task()


    async def stop(self, frame: EndFrame):
        logger.debug(f"{self}: Stop received, signaling sink tasks.")
        if self._sink_clock_queue:
             await self._sink_clock_queue.put((sys.maxsize, frame.id, frame))
        if self._sink_queue:
             await self._sink_queue.put(frame)

        if self._sink_task:
            logger.debug(f"{self}: Waiting for sink task to finish...")
            await self.wait_for_task(self._sink_task)
            self._sink_task = None
            logger.debug(f"{self}: Sink task finished.")
        if self._sink_clock_task:
            logger.debug(f"{self}: Waiting for sink clock task to finish...")
            await self.cancel_task(self._sink_clock_task, timeout=0.5)
            self._sink_clock_task = None
            logger.debug(f"{self}: Sink clock task finished.")

        await self._cancel_camera_task()
        logger.debug(f"{self}: Resetting fade state on stop.")
        self._reset_fade_state()


    async def cancel(self, frame: CancelFrame):
        logger.debug(f"{self}: Cancel received, cancelling tasks and resetting state.")
        await self._cancel_sink_tasks()
        await self._cancel_camera_task()
        self._reset_fade_state()

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame): pass
    async def write_frame_to_camera(self, frame: OutputImageRawFrame): pass
    async def write_raw_audio_frames(self, frames: bytes): pass
    async def send_audio(self, frame: OutputAudioRawFrame): await self.queue_frame(frame, FrameDirection.DOWNSTREAM)
    async def send_image(self, frame: OutputImageRawFrame | SpriteFrame): await self.queue_frame(frame, FrameDirection.DOWNSTREAM)

    def _reset_fade_state(self):
        """Resets all flags related to fading."""
        self._is_fading = False
        self._should_start_fading = False
        self._fade_start_time = 0.0
        self._audio_buffer.clear()


    async def process_frame(self, frame: Frame, direction: FrameDirection):

        if isinstance(frame, StartInterruptionFrame):
            if self.interruptions_allowed:
                if self._is_fading or self._should_start_fading:
                    logger.warning(f"{self}: Ignoring StartInterruptionFrame ({getattr(frame, 'id', 'N/A')}) - fade already active/signaled.")
                    return # Avoid nested fades or redundant signals

                logger.info(f"{self}: StartInterruptionFrame ({getattr(frame, 'id', 'N/A')}) received. Initiating graceful fade.")

                logger.debug(f"{self}: Scheduling immediate UPSTREAM push of BotStoppedSpeakingFrame.")
                asyncio.create_task(
                    self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM),
                    name=f"{self}#UpstreamStopSignal"
                )
                asyncio.create_task(
                    self.push_frame(frame, FrameDirection.UPSTREAM),
                    name=f"{self}#UpstreamInterruptSignal"
                )

                self._should_start_fading = True
                self._audio_buffer.clear() # Clear pre-chunk buffer
                logger.debug(f"{self}: Set _should_start_fading=True, cleared intermediate pre-chunk audio buffer.")

                if self._sink_queue:
                     logger.debug(f"{self}: Putting _FADE_SIGNAL onto sink queue.")
                     await self._sink_queue.put(_FADE_SIGNAL)
                else:
                     logger.error(f"{self}: Sink queue not initialized when signaling fade! Aborting fade signal.")
                     self._should_start_fading = False # Revert signal if queue missing

                await super().process_frame(frame, direction)

                return

            else:
                 logger.debug(f"{self}: Interruptions not allowed, passing StartInterruptionFrame through.")
                 await super().process_frame(frame, direction)
                 await self.push_frame(frame, direction) # Push downstream normally
                 return 
        elif isinstance(frame, StopInterruptionFrame):
             await super().process_frame(frame, direction)
             await self.push_frame(frame, direction)
             return # Stop further processing here

        # This filter applies to frames arriving *after* the StartInterruptionFrame
        # has been processed above.
        if self._should_start_fading or self._is_fading:
            # Allow essential control frames and specific upstream signals
            if isinstance(frame, (StartFrame, CancelFrame, EndFrame, SystemFrame,
                                  MixerControlFrame, TransportMessageFrame, TransportMessageUrgentFrame)):
                logger.trace(f"{self}: Allowing control/system frame {type(frame).__name__} ({getattr(frame, 'id', 'N/A')}) during fade state.")
                # Allow these to pass through for pipeline management
            elif isinstance(frame, BotStoppedSpeakingFrame) and direction == FrameDirection.UPSTREAM:
                 logger.trace(f"{self}: Allowing upstream BotStoppedSpeakingFrame ({getattr(frame, 'id', 'N/A')}) during fade state.")
                 # Allow the specific upstream stop signal through
            elif isinstance(frame, (OutputAudioRawFrame, OutputImageRawFrame, SpriteFrame)) or frame.pts:
                 log_msg = f"Ignoring data/PTS frame {type(frame).__name__}"
                 if hasattr(frame, 'id'):
                     log_msg += f" ({frame.id})"
                 if frame.pts:
                     log_msg += f" with PTS {frame.pts}"
                 log_msg += " during fade signal/active fade."
                 logger.trace(f"{self}: {log_msg}")
                 return # Discard these data frames during fade
            # Allow other upstream frames
            elif direction == FrameDirection.UPSTREAM:
                logger.trace(f"{self}: Allowing other upstream frame {type(frame).__name__} ({getattr(frame, 'id', 'N/A')}) during fade state.")
                pass
            else:
                logger.trace(f"{self}: Ignoring unspecified downstream frame {type(frame).__name__} ({getattr(frame, 'id', 'N/A')}) during fade state.")
                return

        if not isinstance(frame, (StartInterruptionFrame, StopInterruptionFrame)):
             await super().process_frame(frame, direction) # Call super for non-interrupt frames that passed filter

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self.start(frame)
        elif isinstance(frame, CancelFrame):
            # CancelFrame bypasses the fade filter, handle and push.
            await self.cancel(frame) # Call self.cancel first
            await self.push_frame(frame, direction)
        elif isinstance(frame, TransportMessageUrgentFrame):
            # Urgent messages bypass the fade filter.
            await self.send_message(frame)
        elif isinstance(frame, SystemFrame):
            # System frames bypass the fade filter.
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            # EndFrame bypasses the fade filter, handle stop and push.
            await self.stop(frame) # Call self.stop first
            await self.push_frame(frame, direction)
        elif isinstance(frame, MixerControlFrame) and self._params.audio_out_mixer:
            # Mixer controls bypass the fade filter.
            await self._params.audio_out_mixer.process_frame(frame)
        elif isinstance(frame, OutputAudioRawFrame):
            await self._handle_audio(frame)
        elif isinstance(frame, (OutputImageRawFrame, SpriteFrame)):
             await self._handle_image(frame)
        elif frame.pts:
            if self._sink_clock_queue:
                 await self._sink_clock_queue.put((frame.pts, frame.id, frame))
            else:
                 logger.warning(f"{self}: Sink clock queue not initialized when receiving PTS frame.")
        elif direction == FrameDirection.UPSTREAM:
            # Push other allowed upstream frames.
            await self.push_frame(frame, direction)
        else:
             if self._sink_queue:
                 await self._sink_queue.put(frame)
             else:
                  logger.warning(f"{self}: Sink queue not initialized for generic downstream frame {type(frame).__name__}.")
    async def _handle_interruptions_and_reset(self):
        """
        Helper coroutine scheduled by the sink task AFTER fade completion.
        Cancels existing tasks and recreates them (preserving the sink queue)
        to reset the pipeline state cleanly.
        """
        async with self._reset_lock:
            if not self._sink_task and not self._sink_clock_task and not self._camera_out_task:
                logger.warning(f"{self}._handle_interruptions_and_reset: Tasks seem already gone. Skipping reset.")
                return

            logger.critical(f"{self}._handle_interruptions_and_reset: Acquiring lock and resetting tasks.")

            # Cancel existing tasks
            await self._cancel_sink_tasks()
            await self._cancel_camera_task()

            logger.info(f"{self}._handle_interruptions_and_reset: Recreating tasks (preserving sink queue)...")
            self._create_sink_tasks(preserve_queue=True)
            self._create_camera_task()

            logger.info(f"{self}._handle_interruptions_and_reset: Sending final downstream BotStoppedSpeaking.")
            await self._bot_stopped_speaking(force=True, send_downstream_only=True)

            logger.info(f"{self}._handle_interruptions_and_reset: Task reset complete.")


    async def _handle_audio(self, frame: OutputAudioRawFrame):
        """
        Buffers incoming audio, chunks it, and puts chunks onto the sink queue.
        This now runs normally even if _is_fading is True, queuing the *new* audio.
        """

        if not self._params.audio_out_enabled: return
        try:
            resampled = await self._resampler.resample(frame.audio, frame.sample_rate, self._sample_rate)
        except Exception as e:
            logger.error(f"{self}: Resampling failed: {e}")
            return

        cls = type(frame)
        self._audio_buffer.extend(resampled)
        while len(self._audio_buffer) >= self._audio_chunk_size:
            if self._audio_chunk_size <= 0:
                logger.error(f"{self}: Invalid audio chunk size {self._audio_chunk_size}, cannot process.")
                self._audio_buffer.clear(); break

            chunk_bytes = bytes(self._audio_buffer[: self._audio_chunk_size])
            chunk_frame = cls(chunk_bytes, sample_rate=self._sample_rate, num_channels=self._params.audio_out_channels)

            if self._sink_queue:
                await self._sink_queue.put(chunk_frame)
            else:
                logger.warning(f"{self}: Sink queue not available in _handle_audio")

            self._audio_buffer = self._audio_buffer[self._audio_chunk_size :]


    async def _handle_image(self, frame: OutputImageRawFrame | SpriteFrame):
        """Puts image frames onto the correct queue (camera or sink)."""
        if self._should_start_fading or self._is_fading:
            logger.trace(f"{self}: Discarding incoming image frame ({type(frame).__name__} {getattr(frame, 'id', 'N/A')}) due to fade.")
            return

        if not self._params.camera_out_enabled: return
        if self._params.camera_out_is_live:
            if self._camera_out_queue: await self._camera_out_queue.put(frame)
            else: logger.warning(f"{self}: Camera out queue not initialized for live image.")
        else:
            if self._sink_queue: await self._sink_queue.put(frame) # Non-live images go to sink queue
            else: logger.warning(f"{self}: Sink queue not initialized for non-live image.")


    async def _bot_started_speaking(self):
        """Sends BotStartedSpeakingFrame if not already speaking and not fading."""
        if not self._is_fading and not self._should_start_fading and not self._bot_speaking:
            logger.debug(f"{self}: Bot started speaking")
            await self.push_frame(BotStartedSpeakingFrame())
            await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            self._bot_speaking = True


    async def _bot_stopped_speaking(self, force: bool = False, send_downstream_only: bool = False):
        """
        Sends BotStoppedSpeakingFrame downstream and updates internal state.
        """
        if self._bot_speaking or force:
            logger.debug(f"{self}: Bot stopped speaking (Forced: {force}, Downstream Only: {send_downstream_only})")
            await self.push_frame(BotStoppedSpeakingFrame()) # Downstream only signal now
            self._bot_speaking = False
            self._audio_buffer.clear()


    def _create_sink_tasks(self, preserve_queue: bool = False):
        """
        Creates sink tasks. If preserve_queue is True, the existing _sink_queue
        is NOT cleared, allowing queued items (like new audio during fade) to persist.
        """
        logger.debug(f"{self}: Creating sink tasks (Preserve Queue: {preserve_queue})...")
        # Reset fade state flags whenever tasks are recreated
        self._reset_fade_state()

        # Initialize queues if they don't exist
        if self._sink_queue is None:
             self._sink_queue = asyncio.Queue()
             logger.debug(f"{self}: Initialized main sink queue.")
        elif not preserve_queue: # Clear only if not preserving
             if not self._sink_queue.empty():
                 logger.warning(f"{self}: Sink queue not empty ({self._sink_queue.qsize()}) before task creation and preserve=False. Clearing.")
                 while not self._sink_queue.empty():
                     try: self._sink_queue.get_nowait(); self._sink_queue.task_done()
                     except (asyncio.QueueEmpty, ValueError): break
             else:
                 logger.debug(f"{self}: Sink queue exists but is empty (preserve=False).")
        else:
             logger.info(f"{self}: Preserving existing sink queue content ({self._sink_queue.qsize()} items).")


        if self._sink_clock_queue is None:
             self._sink_clock_queue = asyncio.PriorityQueue()
             logger.debug(f"{self}: Initialized sink clock queue.")
        # Always clear clock queue on recreate, as timed items are usually context-specific
        elif not self._sink_clock_queue.empty():
             logger.warning(f"{self}: Sink clock queue not empty ({self._sink_clock_queue.qsize()}) before task creation. Clearing.")
             while not self._sink_clock_queue.empty():
                 try: self._sink_clock_queue.get_nowait(); self._sink_clock_queue.task_done()
                 except (asyncio.QueueEmpty, ValueError): break

        # Create tasks
        if not self._sink_task or self._sink_task.done():
            task_name_suffix = self.__class__.__name__
            self._sink_task = self.create_task(self._sink_task_handler(), name=f"{task_name_suffix}#SinkTask")
            logger.info(f"{self}: Created sink task: {self._sink_task.get_name()}")
        else: logger.warning(f"{self}: Sink task already exists and is running during _create_sink_tasks.")

        if not self._sink_clock_task or self._sink_clock_task.done():
            task_name_suffix = self.__class__.__name__
            self._sink_clock_task = self.create_task(self._sink_clock_task_handler(), name=f"{task_name_suffix}#SinkClockTask")
            logger.info(f"{self}: Created sink clock task: {self._sink_clock_task.get_name()}")
        else: logger.warning(f"{self}: Sink clock task already exists and is running during _create_sink_tasks.")


    async def _cancel_sink_tasks(self):
        """Cancels sink and sink clock tasks if they exist."""
        tasks_to_cancel: List[asyncio.Task] = []
        if self._sink_task:
            task_name = self._sink_task.get_name(); logger.info(f"{self}: Cancelling sink task: {task_name}")
            tasks_to_cancel.append(self._sink_task); self._sink_task = None
        if self._sink_clock_task:
            task_name = self._sink_clock_task.get_name(); logger.info(f"{self}: Cancelling sink clock task: {task_name}")
            tasks_to_cancel.append(self._sink_clock_task); self._sink_clock_task = None

        if tasks_to_cancel:
            results = await asyncio.gather(*(self.cancel_task(task) for task in tasks_to_cancel), return_exceptions=True)
            for i, result in enumerate(results):
                task_name = tasks_to_cancel[i].get_name() if tasks_to_cancel[i] else "UnknownTask"
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    logger.error(f"{self}: Error cancelling task {task_name}: {result}")
            logger.debug(f"{self}: Sink tasks cancellation process complete.")
        else:
            logger.debug(f"{self}: No active sink tasks to cancel.")


    async def _sink_frame_handler(self, frame: Frame):
        """Handles non-audio frames received by the main sink task."""
        if isinstance(frame, OutputImageRawFrame): await self._set_camera_image(frame)
        elif isinstance(frame, SpriteFrame): await self._set_camera_images(frame.images)
        elif isinstance(frame, TransportMessageFrame): await self.send_message(frame)


    async def _sink_clock_task_handler(self):
        # Unchanged from previous correct version
        task_name = asyncio.current_task().get_name() if asyncio.current_task() else "SinkClockTask"
        logger.info(f"{task_name}: Starting loop.")
        running = True
        while running:
            if self._sink_clock_queue is None: logger.warning(f"{task_name}: Clock queue is None, exiting."); break
            try:
                 timestamp, frame_id, frame = await asyncio.wait_for(self._sink_clock_queue.get(), timeout=1.0)
                 logger.trace(f"{task_name}: Dequeued {type(frame).__name__} ({frame_id}) PTS: {timestamp}")
                 if self._should_start_fading or self._is_fading:
                      logger.info(f"{task_name}: Fade signaled/active, stopping PTS processing and requeuing {type(frame).__name__}.")
                      await self._sink_clock_queue.put((timestamp, frame_id, frame)); running = False; break
                 if isinstance(frame, EndFrame):
                      logger.debug(f"{task_name}: Received EndFrame, stopping."); running = False; self._sink_clock_queue.task_done(); break
                 current_time = self.get_clock().get_time()
                 if timestamp > current_time:
                     wait_time = nanoseconds_to_seconds(timestamp - current_time)
                     if wait_time > 0.001: logger.trace(f"{task_name}: Waiting {wait_time:.3f}s"); await asyncio.sleep(wait_time)
                 await self._sink_frame_handler(frame)
                 await self.push_frame(frame)
                 self._sink_clock_queue.task_done()
            except asyncio.TimeoutError: continue
            except asyncio.CancelledError: logger.info(f"{task_name}: Cancelled."); running = False
            except Exception as e: logger.exception(f"{task_name}: Error processing sink clock queue: {e}"); running = False
        logger.info(f"{task_name}: Exiting loop.")


    async def _apply_fade(self, audio_bytes: bytes, start_time: float) -> bytes:
        # Unchanged from previous correct version
        if not audio_bytes or self._taper_duration_s <= 0: return audio_bytes
        elapsed = time.monotonic() - start_time
        progress = min(1.0, max(0.0, elapsed / self._taper_duration_s))
        gain = max(0.0, 1.0 - progress)
        if gain <= 0.001: return b'\x00' * len(audio_bytes)
        if gain >= 0.999: return audio_bytes
        try:
            if len(audio_bytes) % 2 != 0:
                 logger.warning(f"{self}: Odd length audio buffer ({len(audio_bytes)} bytes) in _apply_fade. Trimming."); audio_bytes = audio_bytes[:-1]
                 if not audio_bytes: return b''
            audio_arr = np.frombuffer(audio_bytes, dtype=np.int16)
            faded_arr = (audio_arr.astype(np.float32) * gain).clip(-32768, 32767).astype(np.int16)
            return faded_arr.tobytes()
        except ValueError as e: logger.error(f"{self}: Numpy ValueError applying fade (len: {len(audio_bytes)}): {e}"); return audio_bytes
        except Exception as e: logger.exception(f"{self}: Unexpected error applying fade: {e}"); return audio_bytes

    async def _sink_task_handler(self):
        task_name = asyncio.current_task().get_name() if asyncio.current_task() else "SinkTask"
        logger.info(f"{task_name}: Starting loop.")
        running = True
        local_audio_buffer = bytearray()
        fade_buffer = bytearray()
        fade_start_time = 0.0
        TOTAL_CHUNK_MS = self._params.audio_out_10ms_chunks * 10
        BOT_SPEAKING_CHUNK_PERIOD = max(int(200 / TOTAL_CHUNK_MS), 1) if TOTAL_CHUNK_MS > 0 else 1
        bot_speaking_counter = 0
        last_audio_processed_time = time.monotonic()

        while running:
            processed_item_this_cycle = False
            try:
                if self._should_start_fading:
                    logger.info(f"{task_name}: Detected _should_start_fading=True.")
                    if not self._is_fading:
                        logger.debug(f"{task_name}: Activating fade state.")
                        self._is_fading = True
                        fade_start_time = time.monotonic()
                        self._should_start_fading = False
                        fade_buffer.extend(local_audio_buffer)
                        local_buffer_bytes = len(local_audio_buffer)
                        local_audio_buffer.clear()

                        drained_audio_bytes = 0; items_to_requeue = []
                        if self._sink_queue:
                            logger.debug(f"{task_name}: Draining sink queue into fade buffer...")
                            while not self._sink_queue.empty():
                                try:
                                    queued_item = self._sink_queue.get_nowait()
                                    if queued_item == _FADE_SIGNAL: logger.trace(f"{task_name}: Consumed _FADE_SIGNAL during drain."); self._sink_queue.task_done(); continue
                                    elif isinstance(queued_item, OutputAudioRawFrame):
                                        audio_data = queued_item.audio
                                        if self._params.audio_out_mixer:
                                            try: audio_data = await self._params.audio_out_mixer.mix(audio_data)
                                            except Exception as e: logger.error(f"Audio mixer error during drain: {e}")
                                        fade_buffer.extend(audio_data)
                                        drained_audio_bytes += len(audio_data)
                                    elif isinstance(queued_item, EndFrame): items_to_requeue.append(queued_item); logger.debug(f"{task_name}: EndFrame during drain."); self._sink_queue.task_done(); break
                                    else: logger.trace(f"{task_name}: Requeuing {type(queued_item).__name__} during drain."); items_to_requeue.append(queued_item)
                                    self._sink_queue.task_done()
                                except asyncio.QueueEmpty: break
                                except ValueError: logger.warning(f"{task_name}: ValueError on task_done during drain."); break
                            logger.debug(f"{task_name}: Finished draining queue.")
                            if items_to_requeue: logger.debug(f"{task_name}: Requeuing {len(items_to_requeue)} items."); await asyncio.gather(*(self._sink_queue.put(rq_item) for rq_item in items_to_requeue))

                        total_fade_bytes = len(fade_buffer)
                        logger.info(f"{task_name}: Fade activated. Moved {local_buffer_bytes}b, drained {drained_audio_bytes}b. Total fade: {total_fade_bytes}b.")
                        if total_fade_bytes == 0: logger.warning(f"{task_name}: Fade buffer empty after drain!")
                    else: logger.trace(f"{task_name}: _should_start_fading True but already fading."); self._should_start_fading = False

                if self._is_fading:
                    processed_item_this_cycle = True
                    if len(fade_buffer) > 0:
                        chunk_size = min(self._audio_chunk_size, len(fade_buffer))
                        if chunk_size <= 0: logger.error(f"{task_name}: Invalid chunk size {chunk_size} during fade."); fade_buffer.clear()
                        else:
                            chunk_bytes = bytes(fade_buffer[:chunk_size]); fade_buffer = fade_buffer[chunk_size:]
                            faded_bytes = await self._apply_fade(chunk_bytes, fade_start_time)
                            if len(faded_bytes) > 0: await self.write_raw_audio_frames(faded_bytes); last_audio_processed_time = time.monotonic()
                            chunk_duration_s = len(chunk_bytes) / (self._sample_rate * self._params.audio_out_channels * 2) if self._sample_rate > 0 and self._params.audio_out_channels > 0 else 0.01
                            await asyncio.sleep(max(0.001, chunk_duration_s * 0.8))
                    elapsed_fade = time.monotonic() - fade_start_time
                    if elapsed_fade >= self._taper_duration_s or len(fade_buffer) == 0:
                        log_reason = "Duration reached" if elapsed_fade >= self._taper_duration_s else "Buffer empty"
                        logger.info(f"{task_name}: Fade completed ({log_reason}, Duration: {elapsed_fade:.3f}s).")
                        self._is_fading = False; fade_buffer.clear()
                        logger.info(f"{task_name}: Fade complete, scheduling task reset.")
                        asyncio.create_task(self._handle_interruptions_and_reset())
                    continue

                item = None
                if self._sink_queue is None: logger.error(f"{task_name}: Sink queue is None, exiting."); break
                try: item = await asyncio.wait_for(self._sink_queue.get(), timeout=0.1); processed_item_this_cycle = True
                except asyncio.TimeoutError:
                    if not processed_item_this_cycle and self._bot_speaking and (time.monotonic() - last_audio_processed_time > BOT_VAD_STOP_SECS):
                         logger.debug(f"{task_name}: VAD timeout detected."); await self._bot_stopped_speaking()
                    continue
                except asyncio.QueueEmpty: await asyncio.sleep(0.01); continue

                if item == _FADE_SIGNAL: logger.warning(f"{task_name}: Received _FADE_SIGNAL unexpectedly. Ignoring."); self._sink_queue.task_done(); continue
                if isinstance(item, EndFrame):
                    logger.debug(f"{task_name}: Received EndFrame."); running = False
                    if len(local_audio_buffer) > 0: await self.write_raw_audio_frames(bytes(local_audio_buffer)); local_audio_buffer.clear()
                    await self._bot_stopped_speaking(); self._sink_queue.task_done(); break
                if isinstance(item, OutputAudioRawFrame):
                    frame_audio = item.audio
                    if self._params.audio_out_mixer:
                         try: frame_audio = await self._params.audio_out_mixer.mix(frame_audio)
                         except Exception as e: logger.error(f"Mixer error: {e}");
                    local_audio_buffer.extend(frame_audio)
                    if isinstance(item, TTSAudioRawFrame):
                        await self._bot_started_speaking()
                        if self._bot_speaking:
                            if bot_speaking_counter % BOT_SPEAKING_CHUNK_PERIOD == 0: await self.push_frame(BotSpeakingFrame()); await self.push_frame(BotSpeakingFrame(), FrameDirection.UPSTREAM)
                            bot_speaking_counter += 1
                    while len(local_audio_buffer) >= self._audio_chunk_size:
                        if self._audio_chunk_size <= 0 : logger.error(f"{task_name}: Invalid chunk size."); local_audio_buffer.clear(); break
                        chunk_to_write = bytes(local_audio_buffer[:self._audio_chunk_size]); await self.write_raw_audio_frames(chunk_to_write)
                        last_audio_processed_time = time.monotonic(); local_audio_buffer = local_audio_buffer[self._audio_chunk_size:]
                else: await self._sink_frame_handler(item)
                await self.push_frame(item); self._sink_queue.task_done()
            except asyncio.CancelledError: logger.info(f"{task_name}: Cancelled."); running = False
            except Exception as e:
                logger.exception(f"{task_name}: Error processing item {type(item).__name__ if item else 'None'}: {e}")
                try:
                    if item and self._sink_queue: self._sink_queue.task_done()
                except ValueError: pass
                except Exception as e_td: logger.error(f"Error marking task done after exception: {e_td}")
                running = False
        logger.info(f"{task_name}: Exiting loop. Running: {running}")
        if not asyncio.current_task().cancelled() and len(local_audio_buffer) > 0:
             logger.warning(f"{task_name}: Writing {len(local_audio_buffer)} remaining bytes from local buffer on exit.")
             try: await self.write_raw_audio_frames(bytes(local_audio_buffer))
             except Exception as e: logger.error(f"{task_name}: Error writing final buffer: {e}")
        if self._is_fading: logger.error(f"{task_name}: Exiting loop while _is_fading is still True!")
        if len(fade_buffer) > 0: logger.warning(f"{task_name}: Exiting with {len(fade_buffer)} bytes still in fade buffer!")
        await self._bot_stopped_speaking(force=True)
        self._reset_fade_state(); local_audio_buffer.clear(); fade_buffer.clear()
        logger.info(f"{task_name}: Finished.")


    def _create_camera_task(self):
        if self._params.camera_out_enabled and (not self._camera_out_task or self._camera_out_task.done()):
             if self._camera_out_queue is None: self._camera_out_queue = asyncio.Queue()
             elif not self._camera_out_queue.empty():
                 logger.warning(f"{self}: Camera queue not empty ({self._camera_out_queue.qsize()}) before task creation. Clearing.")
                 while not self._camera_out_queue.empty():
                     try: self._camera_out_queue.get_nowait(); self._camera_out_queue.task_done()
                     except (asyncio.QueueEmpty, ValueError): break
             task_name_suffix = self.__class__.__name__
             self._camera_out_task = self.create_task(self._camera_out_task_handler(), name=f"{task_name_suffix}#CameraTask")
             logger.info(f"{self}: Created camera task: {self._camera_out_task.get_name()}")

    async def _cancel_camera_task(self):
        if self._camera_out_task:
            task_name = self._camera_out_task.get_name(); logger.info(f"{self}: Cancelling camera task: {task_name}")
            await self.cancel_task(self._camera_out_task); self._camera_out_task = None; self._camera_out_queue = None
            logger.debug(f"{self}: Camera task cancelled.")

    async def _draw_image(self, frame: OutputImageRawFrame):
        if not self._params.camera_out_width or not self._params.camera_out_height: logger.error(f"{self}: Camera dimensions not set."); return
        desired_size = (self._params.camera_out_width, self._params.camera_out_height)
        try:
            if not frame.image or not frame.size or not frame.format or frame.size[0] <=0 or frame.size[1] <= 0: logger.error(f"{self}: Invalid image frame: {frame.size}, {frame.format}"); return
            if frame.size != desired_size:
                 logger.warning(f"{self}: Resizing image from {frame.size} to {desired_size}.")
                 image = Image.frombytes(frame.format, frame.size, frame.image); resized_image = image.resize(desired_size)
                 frame = OutputImageRawFrame(resized_image.tobytes(), resized_image.size, resized_image.format)
            await self.write_frame_to_camera(frame)
        except ValueError as e: logger.error(f"{self}: Pillow ValueError drawing image: {e}")
        except Exception as e: logger.exception(f"{self}: Error drawing image: {e}")

    async def _set_camera_image(self, image: OutputImageRawFrame): self._camera_images = itertools.cycle([image])
    async def _set_camera_images(self, images: List[OutputImageRawFrame]): self._camera_images = itertools.cycle(images)

    async def _camera_out_task_handler(self):
        task_name = asyncio.current_task().get_name() if asyncio.current_task() else "CameraTask"
        logger.info(f"{task_name}: Starting loop.")
        self._camera_out_start_time = None; self._camera_out_frame_index = 0
        self._camera_out_frame_duration = 1.0 / self._params.camera_out_framerate if self._params.camera_out_framerate > 0 else 0.1
        self._camera_out_frame_reset = self._camera_out_frame_duration * 5; running = True
        while running:
            try:
                 if self._should_start_fading or self._is_fading: await asyncio.sleep(0.05); continue
                 if self._params.camera_out_is_live:
                    if self._camera_out_queue is None: logger.warning(f"{task_name}: Queue missing."); await asyncio.sleep(0.1); continue
                    try: await asyncio.wait_for(self._camera_out_is_live_handler(), timeout=max(0.1, self._camera_out_frame_duration * 2))
                    except asyncio.TimeoutError: continue
                    except asyncio.QueueEmpty: await asyncio.sleep(self._camera_out_frame_duration / 2); continue
                 elif self._camera_images:
                    try: image = next(self._camera_images); await self._draw_image(image)
                    except StopIteration: logger.info(f"{task_name}: Image iterator exhausted."); self._camera_images = None
                    await asyncio.sleep(self._camera_out_frame_duration)
                 else: await asyncio.sleep(self._camera_out_frame_duration)
            except asyncio.CancelledError: logger.info(f"{task_name}: Cancelled."); running = False
            except Exception as e: logger.exception(f"{task_name}: Error in camera loop: {e}"); await asyncio.sleep(0.5)
        logger.info(f"{task_name}: Exiting loop.")

    async def _camera_out_is_live_handler(self):
        if self._camera_out_queue is None: raise asyncio.QueueEmpty("Camera queue not initialized")
        image = await self._camera_out_queue.get()
        current_time = time.time()
        if self._camera_out_start_time is None: self._camera_out_start_time = current_time; self._camera_out_frame_index = 0
        real_elapsed_time = current_time - self._camera_out_start_time
        target_render_time = self._camera_out_frame_index * self._camera_out_frame_duration
        delay_time = max(0, target_render_time - real_elapsed_time)
        if delay_time > self._camera_out_frame_reset:
             logger.warning(f"{asyncio.current_task().get_name() if asyncio.current_task() else 'Camera'}: Rendering fell behind ({delay_time:.2f}s), resetting timer.")
             self._camera_out_start_time = current_time; self._camera_out_frame_index = 0; delay_time = 0
        if delay_time > 0.001: await asyncio.sleep(delay_time)
        await self._draw_image(image); self._camera_out_frame_index += 1; self._camera_out_queue.task_done()

    async def cleanup(self):
        logger.info(f"{self}: Cleaning up...")
        await self.cancel(CancelFrame())
        await super().cleanup()
        logger.info(f"{self}: Cleanup complete.")
