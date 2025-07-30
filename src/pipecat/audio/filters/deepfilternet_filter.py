
import numpy as np
import torch
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import (
    AudioRawFrame,
    DataFrame,
    Frame,
    OutputAudioRawFrame,
    FilterControlFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
import time
import sys
import warnings

from loguru import logger

warnings.filterwarnings('ignore')

try:
    from df.enhance import enhance, init_df
    from df.io import resample
    DF_AVAILABLE = True
except ImportError as e:
    logger.error(f"Initial DeepFilterNet import failed: {e}")
    logger.error(f"Python executable for this module: {sys.executable}")
    logger.error(f"sys.path for this module: {sys.path}")
    DF_AVAILABLE = False


class DeepFilterNetFilter(BaseAudioFilter):

    def __init__(self,
                 chunk_duration_ms: int = 20,
                 max_buffer_duration_ms: int = 200,
                 device: str = 'cpu',
                 blend_ms: float = 4.0,
                 blend_strength: float = 0.3,
                 enable_postfilter: bool = False,
                 fade_type: str = 'cosine',
                 boundary_smooth_ms: float = 2.0):
        super().__init__()
        if not DF_AVAILABLE:
            raise ImportError(
                "DeepFilterNet not found. Please install it with: pip install deepfilternet")

        if chunk_duration_ms not in [20, 40, 60, 80]:
            raise ValueError("Chunk duration must be 20ms, 40ms, 60ms, or 80ms")
        
        self.chunk_duration_ms = chunk_duration_ms
        self.max_buffer_duration_ms = max_buffer_duration_ms
        self.device = device
        self.blend_ms = blend_ms
        self.blend_strength = blend_strength
        self.enable_postfilter = enable_postfilter
        self.fade_type = fade_type
        self.boundary_smooth_ms = boundary_smooth_ms
        self.sample_rate = 0 # Will be set in start()

        # Add timing for 1-second interval logging
        self.last_log_time = 0
        self.log_interval = 1.0  # 1 second
        self.recent_latencies = []
        self.recent_rms_before = []
        self.recent_rms_after = []

        logger.debug("Loading DeepFilterNet model...")
        self.model, self.df_state, _ = init_df(post_filter=enable_postfilter)
        self.model.eval()

        self.model_sr = self.df_state.sr()
        
        self.chunk_samples = int(chunk_duration_ms * self.model_sr / 1000)
        self.blend_samples = int(blend_ms * self.model_sr / 1000)
        self.boundary_smooth_samples = int(boundary_smooth_ms * self.model_sr / 1000)

        logger.debug(f"Loaded DeepFilterNet - model_sr: {self.model_sr}, device: {device}")
        
        self.reset()

    async def start(self, sample_rate: int):
        logger.debug(f"Starting DeepFilterNet filter with sample rate: {sample_rate}")
        self.sample_rate = sample_rate
        self.last_log_time = time.time()  # Reset log timer

    async def stop(self):
        logger.debug("Stopping DeepFilterNet filter")
        pass

    async def process_frame(self, frame: FilterControlFrame):
        # This is for handling control frames to, for example,
        # enable/disable the filter at runtime.
        pass

    async def filter(self, audio: bytes) -> bytes:
        start_time = time.time()
        if self.sample_rate == 0:
            # This should not happen with BaseAudioFilter, but as a safeguard:
            logger.warning("DeepFilterNetFilter received audio before sample rate was set.")
            return audio

        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        rms_before = np.sqrt(np.mean(audio_np**2))

        if self.sample_rate != self.model_sr:
            audio_tensor = torch.from_numpy(audio_np).float()
            audio_tensor = resample(audio_tensor, self.sample_rate, self.model_sr)
            audio_np = audio_tensor.cpu().numpy()

        # Process audio in chunks
        processed_chunks = []
        for i in range(0, len(audio_np), self.chunk_samples):
            chunk = audio_np[i:i + self.chunk_samples]
            
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)), mode='constant')
            
            processed_chunk = self.process_chunk(chunk)
            
            if i + self.chunk_samples >= len(audio_np):
                actual_length = len(audio_np) - i
                processed_chunk = processed_chunk[:actual_length]

            processed_chunks.append(processed_chunk)
        
        processed_audio = np.concatenate(processed_chunks)

        rms_after = np.sqrt(np.mean(processed_audio**2))

        if self.sample_rate != self.model_sr:
            processed_tensor = torch.from_numpy(processed_audio).float()
            processed_tensor = resample(processed_tensor, self.model_sr, self.sample_rate)
            processed_audio = processed_tensor.cpu().numpy()
        
        latency = (time.time() - start_time) * 1000
        
        # Collect metrics for periodic logging
        self.recent_latencies.append(latency)
        self.recent_rms_before.append(rms_before)
        self.recent_rms_after.append(rms_after)
        
        # Log every 1 second instead of every call
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            if self.recent_latencies:
                avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)
                avg_rms_before = sum(self.recent_rms_before) / len(self.recent_rms_before)
                avg_rms_after = sum(self.recent_rms_after) / len(self.recent_rms_after)
                count = len(self.recent_latencies)
                
                # Calculate noise reduction percentage
                if avg_rms_before > 0.0:
                    reduction_percent = ((avg_rms_before - avg_rms_after) / avg_rms_before) * 100
                else:
                    reduction_percent = 0.0
                
                logger.debug(f"🔊 DeepFilterNet: {count} chunks in 1s | avg latency: {avg_latency:.2f}ms | noise reduction: {reduction_percent:.1f}%")
                
                # Check for potential issues
                if avg_rms_after == 0.0 and avg_rms_before > 0.0:
                    logger.warning("DeepFilterNet may be zeroing out audio - RMS after processing is 0.0")
                
                # Reset for next interval
                self.recent_latencies.clear()
                self.recent_rms_before.clear()
                self.recent_rms_after.clear()
                self.last_log_time = current_time
        
        return (processed_audio * 32768.0).astype(np.int16).tobytes()

    def reset(self):
        self.context_buffer = []
        self.processed_samples = 0
        self.buffer_filled = False
        self.total_chunks = 0
        self.processing_times = []
        self.previous_chunk_tail = None
        self.chunk_boundaries = []
        
        self.df_state = init_df(post_filter=self.enable_postfilter)[1]

    def _create_fade_windows(self, length: int) -> tuple:
        if self.fade_type == 'cosine':
            t = np.linspace(0, np.pi/2, length)
            fade_in = np.sin(t)
            fade_out = np.cos(t)
        elif self.fade_type == 'hann':
            window = np.hanning(length * 2)
            fade_in = window[length:]
            fade_out = window[:length]
        elif self.fade_type == 'exponential':
            t = np.linspace(0, 1, length)
            fade_in = 1 - np.exp(-5 * t)
            fade_out = np.exp(-5 * t)
        else:
            raise ValueError(f"Unknown fade_type: {self.fade_type}")
        
        fade_sum = fade_in + fade_out
        fade_in = fade_in / fade_sum
        fade_out = fade_out / fade_sum
        
        return fade_in, fade_out

    def _prepare_audio_for_processing(self, audio_chunk: np.ndarray) -> np.ndarray:
        if not self.buffer_filled or not self.context_buffer:
            return audio_chunk.copy()
        
        buffer_arrays = [np.array(buf) for buf in self.context_buffer]
        buffer_arrays.append(audio_chunk)
        context_audio = np.concatenate(buffer_arrays)
        
        return context_audio

    def _extract_and_blend_output(self, processed_audio: np.ndarray, original_chunk_size: int) -> np.ndarray:
        if self.buffer_filled and self.context_buffer:
            context_samples = sum(len(buf) for buf in self.context_buffer)
            extracted_chunk = processed_audio[context_samples:context_samples + original_chunk_size]
        else:
            extracted_chunk = processed_audio[:original_chunk_size]
        
        if self.previous_chunk_tail is not None and self.blend_samples > 0:
            blend_len = min(self.blend_samples, len(extracted_chunk), len(self.previous_chunk_tail))
            
            if blend_len > 0:
                fade_in, fade_out = self._create_fade_windows(blend_len)
                
                blend_region = (
                    extracted_chunk[:blend_len] * (1 - self.blend_strength + self.blend_strength * fade_in) +
                    self.previous_chunk_tail[-blend_len:] * (self.blend_strength * fade_out)
                )
                
                extracted_chunk[:blend_len] = blend_region
        
        if self.blend_samples > 0:
            self.previous_chunk_tail = extracted_chunk.copy()
        
        return extracted_chunk

    def _update_context_buffer(self, audio_chunk: np.ndarray):
        self.context_buffer.append(audio_chunk.copy())
        
        max_buffer_samples = int(self.max_buffer_duration_ms * self.model_sr / 1000)
        total_buffer_samples = sum(len(buf) for buf in self.context_buffer)
        
        if not self.buffer_filled and total_buffer_samples >= max_buffer_samples:
            self.buffer_filled = True
            logger.info(f"Buffer filled to {self.max_buffer_duration_ms}ms")
        
        if self.buffer_filled:
            while total_buffer_samples > max_buffer_samples and len(self.context_buffer) > 1:
                removed = self.context_buffer.pop(0)
                total_buffer_samples -= len(removed)

    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        start_time = time.time()
        
        try:
            if len(audio_chunk) != self.chunk_samples:
                if len(audio_chunk) < self.chunk_samples:
                    audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))
                else:
                    audio_chunk = audio_chunk[:self.chunk_samples]
            
            audio_to_process = self._prepare_audio_for_processing(audio_chunk)
            audio_to_process = np.ascontiguousarray(audio_to_process, dtype=np.float32)
            
            audio_tensor = torch.from_numpy(audio_to_process).float()
            audio_tensor = audio_tensor.to(self.device)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            enhanced = enhance(self.model, self.df_state, audio_tensor)
            
            if enhanced.dim() > 1:
                enhanced_np = enhanced.squeeze(0).cpu().numpy()
            else:
                enhanced_np = enhanced.cpu().numpy()
            
            processed_chunk = self._extract_and_blend_output(enhanced_np, len(audio_chunk))
            self._update_context_buffer(audio_chunk)
            
            boundary_position = self.processed_samples + len(processed_chunk)
            self.chunk_boundaries.append(boundary_position)
            
            self.processed_samples += len(audio_chunk)
            self.total_chunks += 1
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            if self.total_chunks % 50 == 0:
                avg_time = np.mean(self.processing_times[-50:])
                chunk_duration = self.chunk_duration_ms / 1000.0
                rtf = avg_time / chunk_duration
                logger.info(f"Processed {self.total_chunks} chunks, RTF: {rtf:.2f}x")
            
            return processed_chunk
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return audio_chunk 