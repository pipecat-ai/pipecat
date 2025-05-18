#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements Ultravox speech-to-text with a locally-loaded model."""

import json
import os
import time
from typing import AsyncGenerator, List, Optional

import numpy as np
from huggingface_hub import login
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService

try:
    from transformers import AutoTokenizer
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Ultravox, you need to `pip install pipecat-ai[ultravox]`.")
    raise Exception(f"Missing module: {e}")


class AudioBuffer:
    """Buffer to collect audio frames before processing.

    Attributes:
        frames: List of AudioRawFrames to process
        started_at: Timestamp when speech started
        is_processing: Flag to prevent concurrent processing
    """

    def __init__(self):
        self.frames: List[AudioRawFrame] = []
        self.started_at: Optional[float] = None
        self.is_processing: bool = False


class UltravoxModel:
    """Model wrapper for the Ultravox multimodal model.

    This class handles loading and running the Ultravox model for speech-to-text.

    Args:
        model_name: The name or path of the Ultravox model to load

    Attributes:
        model_name: The name of the loaded model
        engine: The vLLM engine for model inference
        tokenizer: The tokenizer for the model
        stop_token_ids: Optional token IDs to stop generation
    """

    def __init__(self, model_name: str = "fixie-ai/ultravox-v0_5-llama-3_1-8b"):
        self.model_name = model_name
        self._initialize_engine()
        self._initialize_tokenizer()
        self.stop_token_ids = None

    def _initialize_engine(self):
        """Initialize the vLLM engine for inference."""
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            trust_remote_code=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def _initialize_tokenizer(self):
        """Initialize the tokenizer for the model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def format_prompt(self, messages: list):
        """Format chat messages into a prompt for the model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            str: Formatted prompt string
        """
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    async def generate(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 100,
        audio: np.ndarray = None,
    ):
        """Generate text from audio input using the model.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            audio: Audio data as numpy array

        Yields:
            str: JSON chunks of the generated response
        """
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, stop_token_ids=self.stop_token_ids
        )

        mm_data = {"audio": audio}
        inputs = {"prompt": self.format_prompt(messages), "multi_modal_data": mm_data}
        results_generator = self.engine.generate(inputs, sampling_params, str(time.time()))

        previous_text = ""
        first_chunk = True

        async for output in results_generator:
            prompt_output = output.outputs
            new_text = prompt_output[0].text[len(previous_text) :]
            previous_text = prompt_output[0].text

            # Construct OpenAI-compatible chunk
            chunk = {
                "id": str(int(time.time() * 1000)),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": None,
                    }
                ],
            }

            # Include the role in the first chunk
            if first_chunk:
                chunk["choices"][0]["delta"]["role"] = "assistant"
                first_chunk = False

            # Add new text to the delta if any
            if new_text:
                chunk["choices"][0]["delta"]["content"] = new_text

            # Capture a finish reason if it's provided
            finish_reason = prompt_output[0].finish_reason or None
            if finish_reason and finish_reason != "none":
                chunk["choices"][0]["finish_reason"] = finish_reason

            yield json.dumps(chunk)


class UltravoxSTTService(AIService):
    """Service to transcribe audio using the Ultravox multimodal model.

    This service collects audio frames and processes them with Ultravox
    to generate text transcriptions.

    Args:
        model_name: The Ultravox model to use (ModelSize enum or string)
        hf_token: Hugging Face token for model access
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens to generate
        **kwargs: Additional arguments passed to AIService

    Attributes:
        model: The UltravoxModel instance
        buffer: Buffer to collect audio frames
        temperature: Temperature for text generation
        max_tokens: Maximum tokens to generate
        _connection_active: Flag indicating if service is active
    """

    def __init__(
        self,
        *,
        model_name: str = "fixie-ai/ultravox-v0_5-llama-3_1-8b",
        hf_token: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Authenticate with Hugging Face if token provided
        if hf_token:
            login(token=hf_token)
        elif os.environ.get("HF_TOKEN"):
            login(token=os.environ.get("HF_TOKEN"))
        else:
            logger.warning("No Hugging Face token provided. Model may not load correctly.")

        # Initialize model
        self._model = UltravoxModel(model_name=model_name)

        # Initialize service state
        self._buffer = AudioBuffer()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._connection_active = False
        self._warm_up_duration_sec = 1

        logger.info(f"Initialized UltravoxSTTService with model: {model_name}")

    async def _warm_up_model(self):
        """Warm up the model with silent audio to improve first inference performance.

        This method generates a short segment of silent audio and runs it through
        the model to ensure the model is fully loaded and optimized for the first
        real inference request.
        """
        logger.info("Warming up Ultravox model with silent audio...")

        # Generate silent audio at 16kHz sample rate
        sample_rate = 16000
        silent_audio = self._generate_silent_audio(sample_rate, self._warm_up_duration_sec)

        try:
            # Process the silent audio with the model
            messages = [{"role": "user", "content": "<|audio|>\n"}]
            warmup_generator = self._model.generate(
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                audio=silent_audio,
            )

            # Consume the generator to actually run the inference
            async for _ in warmup_generator:
                pass

            logger.info("Model warm-up completed successfully")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def _generate_silent_audio(self, sample_rate=16000, duration_sec=1.0):
        """Generate silent audio as a numpy array.

        Args:
            sample_rate: Sample rate in Hz
            duration_sec: Duration of silence in seconds

        Returns:
            np.ndarray: Float32 array of zeros representing silent audio
        """
        # Calculate number of samples
        num_samples = int(sample_rate * duration_sec)

        # Create silent audio as float32 in the [-1.0, 1.0] range
        silent_audio = np.zeros(num_samples, dtype=np.float32)

        logger.info(f"Generated {duration_sec}s of silent audio ({num_samples} samples)")
        return silent_audio

    def can_generate_metrics(self) -> bool:
        """Indicates whether this service can generate metrics.

        Returns:
            bool: True, as this service supports metric generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Handle service start.

        Args:
            frame: StartFrame that triggered this method
        """
        await super().start(frame)
        self._connection_active = True

        await self._warm_up_model()

        logger.info("UltravoxSTTService started")

    async def stop(self, frame: EndFrame):
        """Handle service stop.

        Args:
            frame: EndFrame that triggered this method
        """
        await super().stop(frame)
        self._connection_active = False
        logger.info("UltravoxSTTService stopped")

    async def cancel(self, frame: CancelFrame):
        """Handle service cancellation.

        Args:
            frame: CancelFrame that triggered this method
        """
        await super().cancel(frame)
        self._connection_active = False
        self._buffer = AudioBuffer()
        logger.info("UltravoxSTTService cancelled")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames.

        This method collects audio frames and processes them when speech ends.

        Args:
            frame: The frame to process
            direction: Direction of the frame (input/output)
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.info("Speech started")
            self._buffer = AudioBuffer()
            self._buffer.started_at = time.time()

        elif isinstance(frame, AudioRawFrame) and self._buffer.started_at is not None:
            self._buffer.frames.append(frame)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._buffer.frames and not self._buffer.is_processing:
                logger.info("Speech ended, processing buffer...")
                await self.process_generator(self._process_audio_buffer())
                return  # Return early to avoid pushing None frame

        # Only push the original frame if we haven't processed audio
        if frame is not None:
            await self.push_frame(frame, direction)

    async def _process_audio_buffer(self) -> AsyncGenerator[Frame, None]:
        """Process collected audio frames with Ultravox.

        This method concatenates audio frames, processes them with the model,
        and yields the resulting text frames.

        Yields:
            Frame: TextFrame containing the transcribed text
        """
        try:
            self._buffer.is_processing = True

            # Check if we have valid frames before processing
            if not self._buffer.frames:
                logger.warning("No audio frames to process")
                yield ErrorFrame("No audio frames to process")
                return

            # Process audio frames
            audio_arrays = []
            for f in self._buffer.frames:
                if hasattr(f, "audio") and f.audio:
                    # Handle bytes data - these are int16 PCM samples
                    if isinstance(f.audio, bytes):
                        try:
                            # Convert bytes to int16 array
                            arr = np.frombuffer(f.audio, dtype=np.int16)
                            if arr.size > 0:  # Check if array is not empty
                                audio_arrays.append(arr)
                        except Exception as e:
                            logger.error(f"Error processing bytes audio frame: {e}")
                    # Handle numpy array data
                    elif isinstance(f.audio, np.ndarray):
                        if f.audio.size > 0:  # Check if array is not empty
                            # Ensure it's int16 data
                            if f.audio.dtype != np.int16:
                                logger.info(f"Converting array from {f.audio.dtype} to int16")
                                audio_arrays.append(f.audio.astype(np.int16))
                            else:
                                audio_arrays.append(f.audio)

            # Only proceed if we have valid audio arrays
            if not audio_arrays:
                logger.warning("No valid audio data found in frames")
                yield ErrorFrame("No valid audio data found in frames")
                return

            # Concatenate audio frames - all should be int16 now
            audio_data = np.concatenate(audio_arrays)

            audio_int16 = audio_data  # Already in int16 format
            # Save int16 audio

            # Convert int16 to float32 and normalize for model input
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # Generate text using the model
            if self._model:
                try:
                    logger.info("Generating text from audio using model...")

                    # Start metrics tracking
                    await self.start_ttfb_metrics()
                    await self.start_processing_metrics()

                    yield LLMFullResponseStartFrame()

                    async for response in self._model.generate(
                        messages=[{"role": "user", "content": "<|audio|>\n"}],
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        audio=audio_float32,
                    ):
                        # Stop TTFB metrics after first response
                        await self.stop_ttfb_metrics()

                        chunk = json.loads(response)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta:
                                new_text = delta["content"]
                                if new_text:
                                    yield LLMTextFrame(text=new_text)

                    # Stop processing metrics after completion
                    await self.stop_processing_metrics()

                    yield LLMFullResponseEndFrame()

                except Exception as e:
                    logger.error(f"Error generating text from model: {e}")
                    yield ErrorFrame(f"Error generating text: {str(e)}")
            else:
                logger.warning("No model available for text generation")
                yield ErrorFrame("No model available for text generation")

        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}")
            import traceback

            logger.error(traceback.format_exc())
            yield ErrorFrame(f"Error processing audio: {str(e)}")
        finally:
            self._buffer.is_processing = False
            self._buffer.frames = []
            self._buffer.started_at = None
