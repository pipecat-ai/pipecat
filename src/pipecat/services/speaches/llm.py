"""Speaches LLM Service — uses an OpenAI-compatible /v1/chat/completions endpoint."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService


def _merge_consecutive_roles(messages: list) -> list:
    """Collapse consecutive same-role user/assistant messages into one.

    Strict OpenAI-compatible servers — notably vLLM serving Gemma — require the
    conversation to alternate user/assistant/user/assistant and reject the ENTIRE
    request with HTTP 400 ("Conversation roles must alternate ...") otherwise. Live
    voice calls routinely produce two same-role turns in a row: a barge-in, two quick
    user utterances, or a turn whose assistant reply was never recorded. Gemini
    tolerated this; Gemma does not, and the rejected call silently breaks the
    conversation (the bot stalls or repeats a stale answer). Merging consecutive
    same-role messages guarantees alternation without dropping any content.

    System/tool messages, and assistant messages carrying ``tool_calls``, are never
    merged (that could corrupt tool-call flows). The input is not mutated — kept
    messages are shallow-copied before their content is ever combined.
    """
    merged: list = []
    for msg in messages:
        role = msg.get("role")
        prev = merged[-1] if merged else None
        can_merge = (
            role in ("user", "assistant")
            and prev is not None
            and prev.get("role") == role
            and not msg.get("tool_calls")
            and not prev.get("tool_calls")
            and isinstance(msg.get("content"), (str, list))
            and isinstance(prev.get("content"), (str, list))
        )
        if not can_merge:
            merged.append(dict(msg))
            continue
        prev_content, cur_content = prev["content"], msg["content"]
        if isinstance(prev_content, str) and isinstance(cur_content, str):
            prev["content"] = f"{prev_content}\n{cur_content}" if cur_content else prev_content
        else:
            prev_list = (
                prev_content if isinstance(prev_content, list)
                else [{"type": "text", "text": prev_content}]
            )
            cur_list = (
                cur_content if isinstance(cur_content, list)
                else [{"type": "text", "text": cur_content}]
            )
            prev["content"] = prev_list + cur_list
    return merged


@dataclass
class SpeachesLLMSettings(OpenAILLMSettings):
    """Settings for Speaches LLM service."""

    pass


class SpeachesLLMService(OpenAILLMService):
    """Speaches LLM service using an OpenAI-compatible chat completions endpoint."""

    Settings = SpeachesLLMSettings

    def __init__(
        self,
        *,
        api_key: str = "none",
        base_url: str = "http://localhost:11434/v1",
        settings: SpeachesLLMSettings | None = None,
        **kwargs,
    ):
        """Initialize the Speaches LLM service.

        Args:
            api_key: API key for authentication.
            base_url: Base URL of the Speaches-compatible endpoint.
            settings: Optional service settings.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=settings,
            **kwargs,
        )

    def build_chat_completion_params(self, params_from_context) -> dict:
        """Build request params, enforcing strict user/assistant role alternation.

        vLLM/Gemma returns HTTP 400 on non-alternating histories (see
        ``_merge_consecutive_roles``). Overriding here covers BOTH streaming
        completions and ``run_inference`` — they both build their request through
        this method.
        """
        params = super().build_chat_completion_params(params_from_context)
        messages = params.get("messages")
        if isinstance(messages, list) and len(messages) > 1:
            merged = _merge_consecutive_roles(messages)
            if len(merged) != len(messages):
                logger.debug(
                    f"{self}: collapsed {len(messages) - len(merged)} consecutive "
                    f"same-role message(s) for strict role alternation"
                )
            params["messages"] = merged
        return params
