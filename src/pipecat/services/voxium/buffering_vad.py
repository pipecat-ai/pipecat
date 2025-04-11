# buffering_vad.py (Corrected based on VADAnalyzer source)

from loguru import logger
from typing import Optional

# Import VADParams and defaults from the base analyzer module
from pipecat.audio.vad.vad_analyzer import (
    VADAnalyzer, VADState, VADParams,
        VAD_CONFIDENCE, VAD_MIN_VOLUME,
        VAD_START_SECS, VAD_STOP_SECS # Assuming these names based on VADParams
)
from pipecat.audio.vad.silero import SileroVADAnalyzer

class BufferingSileroVADAnalyzer(VADAnalyzer):
    """
    A wrapper around SileroVADAnalyzer that buffers audio chunks within
    voice_confidence to ensure the underlying model receives chunks of
    at least min_chunk_samples.
    """

    def __init__(self, min_chunk_samples: int = 512, **kwargs):
        """
        Initializes the wrapper and the internal SileroVADAnalyzer.

        Args:
            min_chunk_samples (int): The minimum number of samples the underlying
                                     Silero model requires for voice_confidence.
            **kwargs: VAD parameters like sample_rate, confidence, start_secs,
                      stop_secs, min_volume.
        """
        # 1. Determine parameters, using defaults from VADAnalyzer
        sample_rate = kwargs.get("sample_rate", 16000)
        confidence = kwargs.get("confidence", VAD_CONFIDENCE)
        start_secs = kwargs.get("start_secs", VAD_START_SECS)
        stop_secs = kwargs.get("stop_secs", VAD_STOP_SECS)
        min_volume = kwargs.get("min_volume", VAD_MIN_VOLUME)

        logger.info(
            f"Initializing BufferingSileroVADAnalyzer "
            f"(min_model_samples: {min_chunk_samples}, rate: {sample_rate}, "
            f"confidence: {confidence}, start: {start_secs}, "
            f"stop: {stop_secs}, min_vol: {min_volume})"
        )

        # 2. Create the VADParams object for the base class
        vad_params = VADParams(
            confidence=confidence,
            start_secs=start_secs,
            stop_secs=stop_secs,
            min_volume=min_volume
        )

        # 3. Initialize the internal Silero VAD instance
        # Pass only kwargs relevant to SileroVADAnalyzer if its __init__ is different,
        # but likely it just takes them via kwargs too.
        self._internal_vad = SileroVADAnalyzer(**kwargs)
        logger.debug(f"Internal SileroVADAnalyzer created.")
        # We trust internal VAD initializes its sample rate correctly now via kwargs

        # 4. Initialize the base VADAnalyzer class correctly
        super().__init__(sample_rate=sample_rate, params=vad_params)
        logger.debug("VADAnalyzer base class initialized.")

        # 5. Store wrapper-specific state for buffering *inside* voice_confidence
        self._min_model_input_samples = min_chunk_samples
         # Assuming 16-bit audio (2 bytes per sample)
        self._min_model_input_bytes = self._min_model_input_samples * 2
        self._confidence_buffer = bytearray()
        self._last_confidence = 0.0 # Store last valid confidence

    # --- Implement Abstract Methods by Delegating (with buffering for voice_confidence) ---

    def num_frames_required(self) -> int:
        """Delegates to internal VAD. Determines chunk size for analyze_audio."""
        # This defines the chunk size the base analyze_audio loop uses
        return self._internal_vad.num_frames_required() # Must call if it's a method

    def voice_confidence(self, buffer: bytes) -> float:
        """
        Buffers audio chunks before calling the internal VAD's voice_confidence
        to prevent errors with chunks smaller than the model requires.
        """
        self._confidence_buffer.extend(buffer)

        if len(self._confidence_buffer) >= self._min_model_input_bytes:
            chunk_for_model = self._confidence_buffer[:self._min_model_input_bytes]
            self._confidence_buffer = self._confidence_buffer[self._min_model_input_bytes:]

            try:
                # --- ADD LOGGING ---
                internal_rate = self._internal_vad.sample_rate
                logger.debug(f"Calling internal voice_confidence. Internal VAD sample rate: {internal_rate}. Chunk size: {len(chunk_for_model)} bytes.")
                if internal_rate not in [8000, 16000]:
                     # If the rate is invalid here, the internal VAD state is wrong!
                     logger.error(f"!!! INTERNAL VAD SAMPLE RATE IS INVALID ({internal_rate}) BEFORE voice_confidence CALL !!!")
                     # Return last known good confidence to avoid crash, but flag the error
                     return self._last_confidence
                # --- END LOGGING ---

                # Call the internal VAD's confidence function with sufficient data
                self._last_confidence = self._internal_vad.voice_confidence(bytes(chunk_for_model))

            except Exception as e:
                 # Log the specific error from the internal call
                 logger.error(f"Error calling internal voice_confidence: {e}", exc_info=True) # Add exc_info
                 # Return last known good value or 0?
                 return self._last_confidence # Return previous value on error

        return self._last_confidence
    # --- Inherited methods (set_sample_rate, set_params, analyze_audio) ---
    # We DO NOT override set_params anymore.
    # We DO NOT override analyze_audio.
    # We DO NOT need to override sample_rate property (base class handles it)
    # We DO NOT need to override reset_state (base class likely sufficient, unless buffer needs clearing - ADD IT)

    def reset_state(self):
        # Override to clear wrapper buffer and delegate
        logger.debug("Resetting state in BufferingSileroVADAnalyzer.")
        super().reset_state() # Call base class reset if needed (might not exist)
        self._internal_vad.reset_state() # Reset internal VAD
        self._confidence_buffer.clear() # Clear wrapper buffer
        self._last_confidence = 0.0