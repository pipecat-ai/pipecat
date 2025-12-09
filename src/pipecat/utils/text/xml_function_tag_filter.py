#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""XML function tag text filter for TTS services.

Text filter that removes XML function tag markup from text that is
input to TTS services, preventing the TTS from speaking XML tag
syntax aloud.
"""

import re
from typing import Any, Mapping, Optional

from loguru import logger
from pipecat.utils.text.base_text_filter import BaseTextFilter


class XMLFunctionTagFilter(BaseTextFilter):
    """A text filter that removes XML function tag markup from text.
    
    Specifically targets XML-style function tag markup that LLMs
    sometimes include in their responses, such as:
    - <function=always_move_to_main_agenda></function>
    - <function=end_call></function>
    - <function=schedule_interview>{"date": "tomorrow"}</function>
    - <function=safe_calculator({"expression": "two thousand + five hundred"}></function>
    
    The XML function tag markup is removed while preserving the rest of the text content.
    This filter is designed to be used with TTS services via the text_filters parameter.
    """

    def __init__(
        self,
        *,
        custom_patterns: Optional[list[str]] = None,
        preserve_whitespace: bool = True,
    ):
        """Initialize the XML function tag filter.
        
        Args:
            custom_patterns: Optional list of additional regex patterns to filter.
            preserve_whitespace: Whether to preserve surrounding whitespace when removing tags.
        """
        self._preserve_whitespace = preserve_whitespace

        self._default_patterns = [
            # Self-closing function tags: <function=name/> or <function=name />
            r'<function=[^/>]*\s*/\s*>',
            
            # Standard function tags: <function=...>...</function>
            r'<function=.*?</function>',
        ]
        
        # Combine default patterns with custom ones
        all_patterns = self._default_patterns.copy()
        if custom_patterns:
            all_patterns.extend(custom_patterns)
        
        # Compile patterns for performance
        flags = re.IGNORECASE | re.DOTALL
        self._compiled_patterns = [re.compile(pattern, flags) for pattern in all_patterns]

    async def update_settings(self, settings: Mapping[str, Any]):
        """Update the filter's configuration settings.
        
        Supported settings:
            custom_patterns: List of additional regex patterns to filter.
            preserve_whitespace: Whether to preserve surrounding whitespace.
        
        Args:
            settings: Dictionary of setting names to values for configuration.
        """
        # Update preserve_whitespace setting
        if "preserve_whitespace" in settings:
            self._preserve_whitespace = settings["preserve_whitespace"]
        
        # Update patterns if custom_patterns provided
        if "custom_patterns" in settings:
            custom_patterns = settings["custom_patterns"]
            # Recompile patterns with new custom patterns (if any)
            all_patterns = self._default_patterns.copy()
            if custom_patterns:
                all_patterns.extend(custom_patterns)
            
            flags = re.IGNORECASE | re.DOTALL
            self._compiled_patterns = [re.compile(pattern, flags) for pattern in all_patterns]

    async def filter(self, text: str) -> str:
        """Apply filtering transformations to remove XML function tag markup.
        
        Args:
            text: The input text that may contain XML function tag markup.
            
        Returns:
            The filtered text with XML function tag markup removed.
        """
        original_text = text
        filtered_text = text
        removed_tags = []
        
        # Track what XML function tags are being removed
        for i, pattern in enumerate(self._compiled_patterns):
            matches = pattern.findall(filtered_text)
            if matches:
                removed_tags.extend(matches)
                
            if self._preserve_whitespace:
                filtered_text = pattern.sub(' ', filtered_text)
            else:
                filtered_text = pattern.sub('', filtered_text)
        
        # Normalize whitespace
        filtered_text = re.sub(r'\s+', ' ', filtered_text)
        final_text = filtered_text.strip()
        
        # Log filtering of tags
        if removed_tags:
            logger.info(f"XMLFunctionTagFilter: Removed {len(removed_tags)} XML function tag(s)")
            for tag in removed_tags:
                logger.info(f"Removed tag: {tag}")
            logger.info(f"Original text: {repr(original_text)}")
            logger.info(f"Filtered text: {repr(final_text)}")
            logger.info(f"TTS will speak: {final_text}")
        else:
            if final_text.strip():
                logger.debug(f"XMLFunctionTagFilter: No XML function tags found in text: {repr(final_text[:100])}...")
        
        return final_text

    async def handle_interruption(self):
        """Handle interruption events in the processing pipeline.
        
        For this filter, no special interruption handling is needed as it's stateless.
        """
        pass  # No state to reset for this stateless filter

    async def reset_interruption(self):
        """Reset the filter state after an interruption has been handled.
        
        For this filter, no special reset is needed as it's stateless.
        """
        pass  # No state to reset for this stateless filter