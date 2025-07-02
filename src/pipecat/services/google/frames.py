#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google AI service frames for search and grounding functionality.

This module defines specialized frame types for handling search results
and grounding metadata from Google AI models, particularly for Gemini
models that support web search and fact grounding capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from pipecat.frames.frames import DataFrame


@dataclass
class LLMSearchResult:
    """Represents a single search result with confidence scores.

    Parameters:
        text: The search result text content.
        confidence: List of confidence scores associated with the result.
    """

    text: str
    confidence: List[float] = field(default_factory=list)


@dataclass
class LLMSearchOrigin:
    """Represents the origin source of search results.

    Parameters:
        site_uri: URI of the source website.
        site_title: Title of the source website.
        results: List of search results from this origin.
    """

    site_uri: Optional[str] = None
    site_title: Optional[str] = None
    results: List[LLMSearchResult] = field(default_factory=list)


@dataclass
class LLMSearchResponseFrame(DataFrame):
    """Frame containing search results and grounding information from Google AI models.

    This frame is used to convey search results and grounding metadata
    from Google AI models that support web search capabilities. It includes
    the search result text, rendered content, and detailed origin information
    with confidence scores.

    Parameters:
        search_result: The main search result text.
        rendered_content: Rendered content from the search entry point.
        origins: List of search result origins with detailed information.
    """

    search_result: Optional[str] = None
    rendered_content: Optional[str] = None
    origins: List[LLMSearchOrigin] = field(default_factory=list)

    def __str__(self):
        """Return string representation of the search response frame.

        Returns:
            String representation showing search result and origins.
        """
        return f"LLMSearchResponseFrame(search_result={self.search_result}, origins={self.origins})"
