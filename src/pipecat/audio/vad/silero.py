#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time
from typing import Optional

import numpy as np
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

# How often should we reset internal model state
_MODEL_RESET_STATES_TIME = 5.0

try:
    import onnxruntime

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Silero VAD, you need to `pip install pipecat-ai[silero]`.")
    raise Exception(f"Missing module(s): {e}")


class SileroOnnxModel:
    def __init__(self, path, force_onnx_cpu=True):
        import numpy as np

        global np

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if force_onnx_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"], sess_options=opts
            )
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.reset_states()
        self.sample_rates = [8000, 16000]

    def _validate_input(self, x, sr: int):
        if np.ndim(x) == 1:
            x = np.expand_dims(x, 0)
        if np.ndim(x) > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr not in self.sample_rates:
            raise ValueError(
                f"Supported sampling rates: {self.sample_rates} (or multiple of 16000)"
            )
        if sr / np.shape(x)[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        self._state = np.zeros((2, batch_size, 128), dtype="float32")
        self._context = np.zeros((batch_size, 0), dtype="float32")
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if np.shape(x)[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {np.shape(x)[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )

        batch_size = np.shape(x)[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not np.shape(self._context)[1]:
            self._context = np.zeros((batch_size, context_size), dtype="float32")

        x = np.concatenate((self._context, x), axis=1)

        if sr in [8000, 16000]:
            ort_inputs = {"input": x, "state": self._state, "sr": np.array(sr, dtype="int64")}
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs
            self._state = state
        else:
            raise ValueError()

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        return out


class SileroVADAnalyzer(VADAnalyzer):
    def __init__(self, *, sample_rate: Optional[int] = None, params: Optional[VADParams] = None):
        super().__init__(sample_rate=sample_rate, params=params)

        logger.debug("Loading Silero VAD model...")

        model_name = "silero_vad.onnx"
        package_path = "pipecat.audio.vad.data"

        try:
            import importlib_resources as impresources

            model_file_path = str(impresources.files(package_path).joinpath(model_name))
        except BaseException:
            from importlib import resources as impresources

            try:
                with impresources.path(package_path, model_name) as f:
                    model_file_path = f
            except BaseException:
                model_file_path = str(impresources.files(package_path).joinpath(model_name))

        self._model = SileroOnnxModel(model_file_path, force_onnx_cpu=True)

        self._last_reset_time = 0

        logger.debug("Loaded Silero VAD")

    #
    # VADAnalyzer
    #

    def set_sample_rate(self, sample_rate: int):
        if sample_rate != 16000 and sample_rate != 8000:
            raise ValueError(
                f"Silero VAD sample rate needs to be 16000 or 8000 (sample rate: {sample_rate})"
            )

        super().set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        return 512 if self.sample_rate == 16000 else 256

    def voice_confidence(self, buffer) -> float:
        try:
            audio_int16 = np.frombuffer(buffer, np.int16)
            # Divide by 32768 because we have signed 16-bit data.
            audio_float32 = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / 32768.0
            new_confidence = self._model(audio_float32, self.sample_rate)[0]

            # We need to reset the model from time to time because it doesn't
            # really need all the data and memory will keep growing otherwise.
            curr_time = time.time()
            diff_time = curr_time - self._last_reset_time
            if diff_time >= _MODEL_RESET_STATES_TIME:
                self._model.reset_states()
                self._last_reset_time = curr_time

            return new_confidence
        except Exception as e:
            # This comes from an empty audio array
            logger.error(f"Error analyzing audio with Silero VAD: {e}")
            return 0
