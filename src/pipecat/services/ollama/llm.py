#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.services.base_openai import BaseOpenAILLMService


class OLLamaLLMService(BaseOpenAILLMService):
    def __init__(self, *, model: str = "llama2", base_url: str = "http://localhost:11434/v1"):
        super().__init__(model=model, base_url=base_url, api_key="ollama")
