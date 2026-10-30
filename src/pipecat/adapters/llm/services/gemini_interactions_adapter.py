#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Interactions API adapter for Pipecat."""

import base64
import copy
import json
from typing import TYPE_CHECKING, Any, TypedDict, cast
from urllib.parse import urlsplit

from loguru import logger

from pipecat.adapters.llm.base_llm_adapter import BaseLLMAdapter, LLMContextConversionError
from pipecat.adapters.llm.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.adapters.llm.services.gemini_adapter import gemini_function_declarations
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMSpecificMessage,
    NotGiven,
)

if TYPE_CHECKING:
    # Imported for type checking only, so this module still imports on SDKs
    # older than google-genai 2.0.0, where the service reports the version.
    from google.genai.interactions import StepParam, ToolParam


class GeminiInteractionsLLMInvocationParams(TypedDict):
    """Context-based parameters for invoking the Gemini Interactions API."""

    system_instruction: str | None
    input: list["StepParam"]
    tools: list["ToolParam"] | NotGiven


class GeminiInteractionsLLMAdapter(BaseLLMAdapter[GeminiInteractionsLLMInvocationParams]):
    """Gemini Interactions API adapter for Pipecat.

    Converts an LLM context into the Interactions API's list of steps:

    - user messages become ``user_input`` steps, and tool results
      ``function_result`` steps
    - assistant messages become ``model_output`` and ``function_call`` steps
    - a stored thought signature becomes a ``thought`` step ahead of the step
      it belongs to

    Thought signatures are read in the form ``GoogleLLMService`` stores them,
    so a context carries them between the two Gemini services.
    """

    def __init__(self):
        """Initialize the adapter."""
        super().__init__()
        # (adaptation, tool name) pairs already reported, so a tool set needing
        # them warns once rather than on every inference.
        self._warned_schema_adaptations: set[tuple[str, str]] = set()
        self._warned_unsupported_specific_message = False

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for Google.

        Shared with :class:`GeminiLLMAdapter`, so the thought signatures either
        Gemini service stores serve both.
        """
        return "google"

    def supports_file_url(self, url: str, mime_type: str) -> bool:
        """The Gemini developer API consumes its own Files API URIs; other URLs must be inlined."""
        hostname = urlsplit(url).hostname or ""
        return url.startswith("https://") and (
            hostname == "googleapis.com" or hostname.endswith(".googleapis.com")
        )

    async def get_llm_invocation_params(
        self,
        context: LLMContext,
        *,
        system_instruction: str | None = None,
    ) -> GeminiInteractionsLLMInvocationParams:
        """Get Interactions API invocation parameters from a universal LLM context.

        The steps always end on the user's side: the API rejects an input that
        ends with a model turn.

        Args:
            context: The LLM context containing messages, tools, etc.
            system_instruction: Optional system instruction from service settings
                or ``run_inference``.

        Returns:
            Dictionary of parameters for the Interactions API.
        """
        await self.prepare_file_content(context)

        messages = list(self.get_messages(context))
        context_system = None
        if messages and not isinstance(messages[0], LLMSpecificMessage):
            context_system = self._extract_initial_system(
                messages, system_instruction=system_instruction
            )

        try:
            steps = self._to_steps(messages)
        except Exception as e:
            raise LLMContextConversionError(e) from e
        if not steps or steps[-1]["type"] in _MODEL_STEP_TYPES:
            steps.append({"type": "user_input", "content": [{"type": "text", "text": "."}]})

        return {
            "system_instruction": self._resolve_system_instruction(
                context_system, system_instruction, discard_context_system=True
            ),
            "input": cast("list[StepParam]", steps),
            # NOTE: LLMContext's tools are guaranteed to be a ToolsSchema (or NOT_GIVEN)
            "tools": cast("list[ToolParam] | NotGiven", self.from_standard_tools(context.tools)),
        }

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> list[dict[str, Any]]:
        """Convert tool schemas to the Interactions API's tool format.

        Each standard tool becomes a ``function`` tool, with its schema adapted
        to the subset Gemini accepts, and custom Gemini tools are included.

        Args:
            tools_schema: The tools schema containing standard and custom tool definitions.

        Returns:
            List of tool definitions formatted for the Interactions API.
        """
        tools: list[dict[str, Any]] = [
            {"type": "function", **declaration}
            for declaration in gemini_function_declarations(
                tools_schema.standard_tools, self._warned_schema_adaptations
            )
        ]
        if tools_schema.custom_tools:
            tools.extend(
                self.to_interactions_tools(tools_schema.custom_tools.get(AdapterType.GEMINI, []))
            )
        return tools

    @staticmethod
    def to_interactions_tools(tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools in Gemini's ``generateContent`` format to the Interactions API's.

        A ``function_declarations`` entry becomes one ``function`` tool per
        declaration, and a built-in tool keyed by its name, such as
        ``{"google_search": {}}``, becomes that tool's ``type``. A tool already
        carrying a ``type`` passes through as it is.

        Args:
            tools: Tools in Gemini's ``generateContent`` format, as dicts.

        Returns:
            The same tools in the Interactions API's format.
        """
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict) or "type" in tool:
                converted.append(tool)
            elif "function_declarations" in tool:
                converted.extend(
                    {"type": "function", **declaration}
                    for declaration in tool["function_declarations"]
                )
            else:
                for name, config in tool.items():
                    converted.append({"type": name, **(config or {})})
        return converted

    def get_messages_for_logging(self, context: LLMContext) -> list[dict[str, Any]]:
        """Get the context's steps in a format ready for logging.

        Media data and thought signatures are replaced with ``"..."``.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of steps in a format ready for logging.
        """
        steps = copy.deepcopy(self._to_steps(self.get_messages(context), resolve_files=False))
        for step in steps:
            if "signature" in step:
                step["signature"] = "..."
            for item in step.get("content", []):
                if "data" in item:
                    item["data"] = "..."
        return steps

    def _to_steps(
        self, messages: list[LLMContextMessage], *, resolve_files: bool = True
    ) -> list[dict[str, Any]]:
        """Convert context messages to steps, thought signatures included."""
        steps: list[dict[str, Any]] = []
        signatures: list[dict[str, Any]] = []
        tool_names: dict[str, str] = {}

        for message in messages:
            if isinstance(message, LLMSpecificMessage):
                specific = message.message
                if isinstance(specific, dict) and specific.get("type") == "thought_signature":
                    signatures.append(specific)
                elif isinstance(specific, dict) and specific.get("type") in _STEP_TYPES:
                    steps.append(specific)
                else:
                    self._warn_unsupported_specific_message()
                continue
            steps.extend(self._message_steps(message, tool_names, resolve_files))

        steps = self._insert_thought_steps(steps, signatures)
        steps = self._regroup_parallel_calls(steps)
        return self._start_model_turns_with_thoughts(steps)

    def _message_steps(
        self, message: Any, tool_names: dict[str, str], resolve_files: bool
    ) -> list[dict[str, Any]]:
        """Convert one standard context message to steps."""
        msg = cast(dict[str, Any], message)
        role = msg["role"]
        content = msg.get("content")

        if role == "tool":
            call_id = msg.get("tool_call_id") or ""
            step: dict[str, Any] = {
                "type": "function_result",
                "call_id": call_id,
                "result": _function_result(content),
            }
            if call_id in tool_names:
                step["name"] = tool_names[call_id]
            return [step]

        items = self._content_items(content, resolve_files)

        if role == "assistant":
            steps = [{"type": "model_output", "content": items}] if items else []
            for call in msg.get("tool_calls") or []:
                function = call["function"]
                tool_names[call["id"]] = function["name"]
                steps.append(
                    {
                        "type": "function_call",
                        "id": call["id"],
                        "name": function["name"],
                        "arguments": json.loads(function["arguments"] or "{}"),
                    }
                )
            return steps

        # The Interactions API has no other roles; a non-initial system or
        # developer message speaks for the user.
        return [{"type": "user_input", "content": items}] if items else []

    def _content_items(self, content: Any, resolve_files: bool) -> list[dict[str, Any]]:
        """Convert a message's content to Interactions content items."""
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []

        items: list[dict[str, Any]] = []
        for c in content:
            kind = c["type"]
            if kind == "text":
                if c["text"]:
                    items.append({"type": "text", "text": c["text"]})
            elif kind == "image_url":
                url = c["image_url"]["url"]
                if url.startswith("data:"):
                    items.append(_data_url_item(url))
                else:
                    items.append({"type": "image", "uri": url})
            elif kind == "input_audio":
                items.append(
                    {"type": "audio", "mime_type": "audio/wav", "data": c["input_audio"]["data"]}
                )
            elif kind == "file_base64":
                items.append(_data_url_item(c["file"]["file_data"], c["file"]["mime_type"]))
            elif kind == "file_url":
                file = c["file"]
                resolved = self.inlined_file_content(file) if resolve_files else None
                if resolved is None:
                    items.append(
                        {
                            "type": _media_type(file["mime_type"]),
                            "mime_type": file["mime_type"],
                            "uri": file["url"],
                        }
                    )
                else:
                    # Base64 adapters cache the data URL form.
                    items.append(_data_url_item(cast(str, resolved), file["mime_type"]))
            elif kind == "file_data":
                file = c["file_data"]
                mime_type = file.get("mime_type") or ""
                item = {"type": _media_type(mime_type), "uri": file.get("file_uri")}
                if mime_type:
                    item["mime_type"] = mime_type
                items.append(item)
        return items

    @classmethod
    def _insert_thought_steps(
        cls, steps: list[dict[str, Any]], signatures: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Put each stored signature, as a thought step, ahead of the step it belongs to.

        A signature's bookmark names the function call, text or image it was
        received with. Signatures are stored in conversation order, so each
        one is matched among the model steps after the previous match.
        """
        thoughts: dict[int, str] = {}
        start = 0
        for entry in signatures:
            signature = entry.get("signature")
            bookmark = entry.get("bookmark")
            if not signature or not bookmark:
                continue
            for i in range(start, len(steps)):
                if steps[i]["type"] in _MODEL_STEP_TYPES and _bookmark_matches(bookmark, steps[i]):
                    thoughts[i] = _encode_signature(signature)
                    start = i + 1
                    break

        if not thoughts:
            return steps
        result = []
        for i, step in enumerate(steps):
            if i in thoughts:
                result.append({"type": "thought", "signature": thoughts[i]})
            result.append(step)
        return result

    @staticmethod
    def _regroup_parallel_calls(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Put a batch of parallel calls back together, ahead of their results.

        The context stores each call with its result right after it, while the
        model made the batch's calls in one turn and signed only the first. So
        a signed call starts a batch, the unsigned calls and the results that
        follow it belong to it, and the next thought step ends it.
        """
        result: list[dict[str, Any]] = []
        i = 0
        while i < len(steps):
            if not (
                steps[i]["type"] == "thought"
                and i + 1 < len(steps)
                and steps[i + 1]["type"] == "function_call"
            ):
                result.append(steps[i])
                i += 1
                continue
            calls = [steps[i]]
            results = []
            i += 1
            while i < len(steps) and steps[i]["type"] in ("function_call", "function_result"):
                (calls if steps[i]["type"] == "function_call" else results).append(steps[i])
                i += 1
            result.extend(calls)
            result.extend(results)
        return result

    @staticmethod
    def _start_model_turns_with_thoughts(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Open each model turn that needs a thought step with one.

        A model turn is a run of steps between the user's. The API rejects a
        model turn holding a thought step that doesn't open with one, which a
        context produces when it stores a model's text and its function calls
        as separate messages, so that turn's first thought step moves to its
        start. It also rejects function calls with no thought step, which is
        how calls from another provider or the application look, so those
        turns open with the placeholder signature Google documents for them.
        """
        result: list[dict[str, Any]] = []
        turn: list[dict[str, Any]] = []

        def flush():
            thought = next((s for s in turn if s["type"] == "thought"), None)
            if thought is not None:
                turn.remove(thought)
                turn.insert(0, thought)
            elif any(s["type"] == "function_call" for s in turn):
                turn.insert(0, {"type": "thought", "signature": _PLACEHOLDER_SIGNATURE})
            result.extend(turn)
            turn.clear()

        for step in steps:
            if step["type"] in _MODEL_STEP_TYPES:
                turn.append(step)
            else:
                flush()
                result.append(step)
        flush()
        return result

    def _warn_unsupported_specific_message(self):
        if self._warned_unsupported_specific_message:
            return
        self._warned_unsupported_specific_message = True
        logger.warning(
            "Skipping Google-specific context messages the Interactions API can't take. "
            "Only thought signatures and Interactions steps are sent."
        )


_MODEL_STEP_TYPES = ("model_output", "function_call", "thought")

_STEP_TYPES = ("user_input", "function_result", *_MODEL_STEP_TYPES)

_PLACEHOLDER_SIGNATURE = base64.b64encode(b"skip_thought_signature_validator").decode()


def _media_type(mime_type: str) -> str:
    """The Interactions content type for a MIME type."""
    for media in ("image", "audio", "video"):
        if mime_type.startswith(f"{media}/"):
            return media
    return "document"


def _data_url_item(url: str, mime_type: str | None = None) -> dict[str, Any]:
    """A content item for a base64 data URL (``data:<mime>;base64,<data>``)."""
    header, data = url.split(",", 1)
    mime_type = mime_type or header.split(":", 1)[1].split(";", 1)[0]
    return {"type": _media_type(mime_type), "mime_type": mime_type, "data": data}


def _function_result(content: Any) -> dict[str, Any] | str:
    """A tool message's content as a function result: a JSON object, or text."""
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return json.dumps(content)
    try:
        decoded = json.loads(content)
    except ValueError:
        return content
    return decoded if isinstance(decoded, dict) else content


def _encode_signature(signature: bytes | str) -> str:
    """Encode a thought signature for the wire.

    Signatures are stored as raw bytes; a string is already in the base64 form
    the API takes.
    """
    if isinstance(signature, bytes):
        return base64.b64encode(signature).decode()
    return signature


def _bookmark_matches(bookmark: dict[str, Any], step: dict[str, Any]) -> bool:
    """Whether a thought signature's bookmark names this step."""
    if call_id := bookmark.get("function_call"):
        return step["type"] == "function_call" and step["id"] == call_id
    if step["type"] != "model_output":
        return False
    if text := bookmark.get("text"):
        step_text = " ".join("".join(i.get("text", "") for i in step["content"]).split())
        text = " ".join(text.split())
        # Either may be a prefix of the other: an interruption cuts the stored
        # text short, and a signature can arrive with the first chunk of a reply.
        return bool(step_text) and (text.startswith(step_text) or step_text.startswith(text))
    if (inline_data := bookmark.get("inline_data")) is not None:
        data = getattr(inline_data, "data", None)
        # Matching in conversation order, the size is enough to tell images apart.
        return data is not None and any(
            "data" in item and len(base64.b64decode(item["data"])) == len(data)
            for item in step["content"]
        )
    return False
