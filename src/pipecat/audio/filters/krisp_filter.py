import os

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    import krisp_audio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the Krisp filter, you need to `pip install pipecat-ai[krisp]`.")
    raise Exception(f"Missing module: {e}")


class KrispProcessorManager:
    """
    Ensures that only one KrispAudioProcessor instance exists for the entire program.
    """

    _krisp_instance = None

    @classmethod
    def get_processor(cls, sample_rate: int, sample_type: str, frame_dur: int, model_path: str, suppression_level: int = 100):
        if cls._krisp_instance is None:
            # Initialize krisp_audio global resources
            krisp_audio.globalInit("")
            
            # Create model info
            model_info = krisp_audio.ModelInfo()
            model_info.path = model_path
            
            # Create session config
            nc_cfg = krisp_audio.NcSessionConfig()
            nc_cfg.inputSampleRate = cls._int_to_sample_rate(sample_rate)
            nc_cfg.inputFrameDuration = cls._int_to_frame_dur(frame_dur)
            nc_cfg.outputSampleRate = nc_cfg.inputSampleRate
            nc_cfg.modelInfo = model_info
            
            # Create the appropriate instance based on sample type
            if sample_type == 'FLOAT':
                cls._krisp_instance = krisp_audio.NcFloat.create(nc_cfg)
            elif sample_type == 'PCM_16':
                cls._krisp_instance = krisp_audio.NcInt16.create(nc_cfg)
            else:
                raise ValueError(f"Unsupported sample type {sample_type}")
                
        return cls._krisp_instance
    
    @classmethod
    def _int_to_sample_rate(cls, sample_rate):
        rates = {
            8000: krisp_audio.SamplingRate.Sr8000Hz,
            16000: krisp_audio.SamplingRate.Sr16000Hz,
            24000: krisp_audio.SamplingRate.Sr24000Hz,
            32000: krisp_audio.SamplingRate.Sr32000Hz,
            44100: krisp_audio.SamplingRate.Sr44100Hz,
            48000: krisp_audio.SamplingRate.Sr48000Hz
        }
        if sample_rate not in rates:
            raise ValueError(f"Unsupported sample rate: {sample_rate}")
        return rates[sample_rate]

    @classmethod
    def _int_to_frame_dur(cls, frame_dur):
        durations = {
            10: krisp_audio.FrameDuration.Fd10ms,
            15: krisp_audio.FrameDuration.Fd15ms,
            20: krisp_audio.FrameDuration.Fd20ms,
            30: krisp_audio.FrameDuration.Fd30ms,
            32: krisp_audio.FrameDuration.Fd32ms
        }
        if frame_dur not in durations:
            raise ValueError(f"Unsupported frame duration: {frame_dur}")
        return durations[frame_dur]


class KrispFilter(BaseAudioFilter):
    def __init__(
        self, 
        sample_type: str = "PCM_16", 
        frame_dur: int = 20, 
        model_path: str = None,
        suppression_level: int = 100
    ) -> None:
        """Initializes the Krisp noise cancellation filter with customizable audio processing settings.

        :param sample_type: The type of audio sample, default is 'PCM_16'.
        :param frame_dur: Frame duration in milliseconds. Supported: 10, 15, 20, 30, 32. Default is 20.
        :param model_path: Path to the Krisp model (.kef file); defaults to environment variable KRISP_MODEL_PATH if not provided.
        :param suppression_level: Noise suppression level in the range [0, 100]. Default is 100 (full noise canceling).
        """
        super().__init__()

        # Set model path, checking environment if not specified
        logger.debug(f"Model path krisp: {model_path}")
        self._model_path = model_path or os.getenv("KRISP_MODEL_PATH")
        if not self._model_path:
            logger.error(
                "Model path for Krisp is not provided and KRISP_MODEL_PATH is not set."
            )
            raise ValueError("Model path for Krisp must be provided.")

        self._sample_type = sample_type
        self._frame_dur = frame_dur
        self._sample_rate = 0
        self._suppression_level = suppression_level
        self._filtering = True
        self._krisp_processor = None

    async def start(self, sample_rate: int):
        self._sample_rate = sample_rate
        self._krisp_processor = KrispProcessorManager.get_processor(
            self._sample_rate, self._sample_type, self._frame_dur, self._model_path
        )

    async def stop(self):
        self._krisp_processor = None
        # Note: We're not calling globalDestroy() here to avoid issues if multiple filters exist
        # This would typically be handled at application shutdown

    async def process_frame(self, frame: FilterControlFrame):
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        if not self._filtering or self._krisp_processor is None:
            return audio

        # Convert bytes to appropriate numpy array based on sample type
        if self._sample_type == 'PCM_16':
            data = np.frombuffer(audio, dtype=np.int16)
        elif self._sample_type == 'FLOAT':
            data = np.frombuffer(audio, dtype=np.float32)
        else:
            logger.error(f"Unsupported sample type: {self._sample_type}")
            return audio

        # Process the audio chunk to reduce noise
        processed_data = self._krisp_processor.process(data, self._suppression_level)
        
        # Convert back to bytes
        return processed_data.tobytes()