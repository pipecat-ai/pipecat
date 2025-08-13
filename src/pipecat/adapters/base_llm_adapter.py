#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base adapter for LLM provider integration.

This module provides the abstract base class for implementing LLM provider-specific
adapters that handle tool format conversion and standardization.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, List, TypeVar, Union, cast

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext

# Should be a TypedDict
TLLMInvocationParams = TypeVar("TLLMInvocationParams", bound=dict[str, Any])


# TODO: fix everywhere we subclass BaseLLMAdapter...
class BaseLLMAdapter(ABC, Generic[TLLMInvocationParams]):
    """Abstract base class for LLM provider adapters.

    Provides a standard interface for converting to provider-specific formats.

    Handles:
    - Extracting provider-specific parameters for LLM invocation from a
      universal LLM context
    - Converting standardized tools schema to provider-specific tool formats.
    - Extracting messages from the LLM context for the purposes of logging
      about the specific provider.

    Subclasses must implement provider-specific conversion logic.
    """

    @abstractmethod
    def get_llm_invocation_params(self, context: LLMContext) -> TLLMInvocationParams:
        """Get provider-specific LLM invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.

        Returns:
            Provider-specific parameters for invoking the LLM.
        """
        pass

    @abstractmethod
    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Any]:
        """Convert tools schema to the provider's specific format.

        Args:
            tools_schema: The standardized tools schema to convert.

        Returns:
            List of tools in the provider's expected format.
        """
        pass

    @abstractmethod
    def get_messages_for_logging(self, context: LLMContext) -> List[dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about this provider.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about this
            provider.
        """
        pass

    # TODO: should this also be able to return NotGiven?
    def from_standard_tools(self, tools: Any) -> List[Any]:
        """Convert tools from standard format to provider format.

        Args:
            tools: Tools in standard format or provider-specific format.

        Returns:
            List of tools converted to provider format, or original tools
            if not in standard format.
        """
        if isinstance(tools, ToolsSchema):
            logger.debug(f"Retrieving the tools using the adapter: {type(self)}")
            return self.to_provider_tools_format(tools)
        # Fallback to return the same tools in case they are not in a standard format
        return tools

    def create_wav_header(self, sample_rate, num_channels, bits_per_sample, data_size):
        """Create a WAV file header for audio data.

        Args:
            sample_rate: Audio sample rate in Hz.
            num_channels: Number of audio channels.
            bits_per_sample: Bits per audio sample.
            data_size: Size of audio data in bytes.

        Returns:
            WAV header as a bytearray.
        """
        # RIFF chunk descriptor
        header = bytearray()
        header.extend(b"RIFF")  # ChunkID
        header.extend((data_size + 36).to_bytes(4, "little"))  # ChunkSize: total size - 8
        header.extend(b"WAVE")  # Format
        # "fmt " sub-chunk
        header.extend(b"fmt ")  # Subchunk1ID
        header.extend((16).to_bytes(4, "little"))  # Subchunk1Size (16 for PCM)
        header.extend((1).to_bytes(2, "little"))  # AudioFormat (1 for PCM)
        header.extend(num_channels.to_bytes(2, "little"))  # NumChannels
        header.extend(sample_rate.to_bytes(4, "little"))  # SampleRate
        # Calculate byte rate and block align
        byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        block_align = num_channels * (bits_per_sample // 8)
        header.extend(byte_rate.to_bytes(4, "little"))  # ByteRate
        header.extend(block_align.to_bytes(2, "little"))  # BlockAlign
        header.extend(bits_per_sample.to_bytes(2, "little"))  # BitsPerSample
        # "data" sub-chunk
        header.extend(b"data")  # Subchunk2ID
        header.extend(data_size.to_bytes(4, "little"))  # Subchunk2Size
        return header

    # TODO: we can move the logic to also handle the Messages here
