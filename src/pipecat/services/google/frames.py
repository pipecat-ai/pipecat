#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import dataclass, field
from typing import List, Optional

from pipecat.frames.frames import DataFrame


@dataclass
class LLMSearchResult:
    text: str
    confidence: List[float] = field(default_factory=list)


@dataclass
class LLMSearchOrigin:
    site_uri: Optional[str] = None
    site_title: Optional[str] = None
    results: List[LLMSearchResult] = field(default_factory=list)


@dataclass
class LLMSearchResponseFrame(DataFrame):
    search_result: Optional[str] = None
    rendered_content: Optional[str] = None
    origins: List[LLMSearchOrigin] = field(default_factory=list)

    def __str__(self):
        return f"LLMSearchResponseFrame(search_result={self.search_result}, origins={self.origins})"
