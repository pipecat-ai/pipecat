#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base adapter for LLM provider integration.

This module provides the abstract base class for implementing LLM provider-specific
adapters that handle tool format conversion and standardization.
"""

import base64
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, TypeVar, cast
from urllib.parse import urlsplit

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMSpecificMessage,
    NotGiven,
)
from pipecat.utils.file_resolver import FileResolver
from pipecat.utils.security.ssrf import UrlReachability

# Should be a TypedDict
TLLMInvocationParams = TypeVar("TLLMInvocationParams", bound=Mapping[str, Any])


class LLMContextConversionError(Exception):
    """Raised when converting a universal ``LLMContext`` to a provider's message format fails.

    Adapters that transform context messages into a provider-specific format
    raise this from their conversion routine, wrapping the underlying error
    (preserved as ``__cause__``). Its message identifies the failure as a
    context-mapping problem and carries the underlying cause. The corresponding
    LLM service catches this and surfaces it in the ``ErrorFrame`` it pushes
    upstream.
    """

    def __init__(self, cause: Exception):
        """Initialize the error.

        Args:
            cause: The underlying exception raised during message conversion.
        """
        super().__init__(f"Error mapping context messages to provider format: {cause}")


class BaseLLMAdapter(ABC, Generic[TLLMInvocationParams]):
    """Abstract base class for LLM provider adapters.

    Provides a standard interface for converting to provider-specific formats.

    Handles:

    - Extracting provider-specific parameters for LLM invocation from a
      universal LLM context
    - Converting standardized tools schema to provider-specific tool formats.
    - Extracting messages from the LLM context for the purposes of logging
      about the specific provider.
    - Resolving conflicts between ``system_instruction`` and initial
      system/developer messages in the conversation context.

    Subclasses must implement provider-specific conversion logic.
    """

    def __init__(self):
        """Initialize the adapter."""
        self._warned_system_instruction = False
        self._warned_context_system_message = False
        self._builtin_tools: dict[str, FunctionSchema] = {}
        self._file_resolver: FileResolver | None = None

    @property
    def builtin_tools(self) -> dict[str, FunctionSchema]:
        """Built-in tools automatically merged into every inference request.

        Keyed by tool name for O(1) lookup, insertion, and removal.  The
        service injects tools here so they are sent transparently on every
        inference request without the user having to add them to their
        ``ToolsSchema``.

        Returns:
            Mutable dict mapping tool name to ``FunctionSchema``.
        """
        return self._builtin_tools

    @property
    @abstractmethod
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for this LLM provider.

        Returns:
            The identifier string.
        """
        pass

    @abstractmethod
    def get_llm_invocation_params(self, context: LLMContext, **kwargs) -> TLLMInvocationParams:
        """Get provider-specific LLM invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.
            **kwargs: Additional provider-specific arguments that subclasses can use.

        Returns:
            Provider-specific parameters for invoking the LLM.
        """
        pass

    @abstractmethod
    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> list[Any]:
        """Convert tools schema to the provider's specific format.

        Args:
            tools_schema: The standardized tools schema to convert.

        Returns:
            List of tools in the provider's expected format.
        """
        pass

    @abstractmethod
    def get_messages_for_logging(self, context: LLMContext) -> list[dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about this provider.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about this
            provider.
        """
        pass

    # Whether this provider's conversion consumes raw bytes (rather than the
    # base64 payload of a data URL) for inline file content. Decides which
    # form the resolution pass prepares in the resolver's cache, so conversion
    # never re-decodes or re-encodes a file on later turns.
    prefers_raw_file_bytes: bool = False

    @property
    def file_resolver(self) -> FileResolver | None:
        """The resolver that fetches and caches file content for this adapter.

        Set once by the owning LLM service. Every conversion this adapter
        performs — invocation params and logging alike — reads the same
        resolver caches. Without one, conversion still passes through URLs the
        provider consumes directly, but fails on any file that would need
        fetching.
        """
        return self._file_resolver

    @file_resolver.setter
    def file_resolver(self, resolver: FileResolver | None) -> None:
        self._file_resolver = resolver

    def supports_file_url(self, url: str, mime_type: str) -> bool:
        """Whether the provider can be handed `url` to fetch itself.

        This covers both URLs the provider fetches over the public internet
        and cloud-storage URIs it resolves through its own IAM (e.g. Bedrock
        reading ``s3://``). A URL the provider can't consume is instead
        fetched into the file resolver's cache and inlined at conversion —
        see :meth:`prepare_llm_invocation_params`.

        Args:
            url: The file URL from a ``file_url`` context content item.
            mime_type: The file's MIME type.

        Returns:
            True if conversion can pass `url` through to the provider.
        """
        return False

    async def prepare_llm_invocation_params(
        self, context: LLMContext, **kwargs
    ) -> TLLMInvocationParams:
        """Resolve the context's file content, then convert it to provider invocation params.

        The one entry point services call per LLM invocation: runs the async
        file resolution (:meth:`prepare_file_content`) into the
        :attr:`file_resolver`'s caches, then delegates to the synchronous
        :meth:`get_llm_invocation_params`, whose conversion reads those caches.
        The context itself is never modified.

        Args:
            context: The LLM context to resolve and convert.
            **kwargs: Forwarded to :meth:`get_llm_invocation_params`.

        Returns:
            Provider-specific parameters for invoking the LLM.

        Raises:
            LLMContextConversionError: If a file can't be resolved or the
                context can't be converted.
        """
        if self._file_resolver is not None:
            await self.prepare_file_content(context)
        return self.get_llm_invocation_params(context, **kwargs)

    async def prepare_file_content(self, context: LLMContext) -> None:
        """Fetch the file content this provider will need into the resolver's caches.

        A ``file_url`` item whose URL the provider consumes directly
        (:meth:`supports_file_url`, and — for ``http(s)`` — publicly routable)
        needs nothing. Any other ``file_url`` item is fetched into the
        :attr:`file_resolver`'s cache, in the form this provider's conversion
        reads: raw bytes when ``prefers_raw_file_bytes``, a base64 data URL
        otherwise. Inline ``file_base64`` items get their decoded bytes cached
        for raw-bytes providers. The resolver caches by URL, so each file is
        fetched once no matter how many turns — or how many adapters sharing
        the resolver — consume it; only the per-provider pass-through decision
        is re-evaluated here on every run.

        Args:
            context: The LLM context whose messages to resolve.

        Raises:
            LLMContextConversionError: If a file URL can't be resolved (refused
                by the reachability policy, download failure, unresolvable
                scheme, corrupt base64). The service handles it like any other
                conversion failure, including invalid-file-message cleanup.
        """
        resolver = self._file_resolver
        if resolver is None:
            return
        for message in context.get_messages():
            if isinstance(message, LLMSpecificMessage):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for raw_item in content:
                if not isinstance(raw_item, dict):
                    continue
                item = cast("dict[str, Any]", raw_item)
                try:
                    if item.get("type") == "file_url":
                        file_data = item["file"]
                        url = file_data["url"]
                        mime_type = file_data["mime_type"]
                        if await self._can_pass_file_url(url, mime_type, resolver):
                            continue
                        if self.prefers_raw_file_bytes:
                            await resolver.fetch(url)
                        else:
                            await resolver.ensure_data_url(url, mime_type)
                    elif item.get("type") == "file_base64" and self.prefers_raw_file_bytes:
                        await resolver.fetch(item["file"]["file_data"])
                except Exception as e:
                    raise LLMContextConversionError(e) from e

    async def _can_pass_file_url(self, url: str, mime_type: str, resolver: FileResolver) -> bool:
        """Whether `url` can be handed to the provider to fetch itself."""
        if not self.supports_file_url(url, mime_type):
            return False
        if urlsplit(url).scheme not in ("http", "https"):
            # A cloud-storage URI resolved via the provider's own IAM;
            # reachability from here is irrelevant.
            return True
        return await resolver.classify(url) == UrlReachability.PUBLIC

    def inlined_file_content(self, file_data: Mapping[str, Any]) -> bytes | str | None:
        """Return the content to inline for a ``file_url`` item, or None when the URL passes through.

        Args:
            file_data: The item's ``file`` mapping (``url``, ``mime_type``, ...).

        Returns:
            None when the URL passes through for the provider to fetch itself;
            otherwise the resolved content from the :attr:`file_resolver`'s
            caches — raw bytes when ``prefers_raw_file_bytes``, a base64 data
            URL otherwise.

        Raises:
            ValueError: If the provider can't consume the URL and no resolved
                content is cached (wrapped as ``LLMContextConversionError`` by
                the conversion routine's caller).
        """
        resolver = self._file_resolver
        url = file_data["url"]
        mime_type = file_data["mime_type"]
        if self.supports_file_url(url, mime_type):
            if urlsplit(url).scheme not in ("http", "https"):
                return None
            reachability = resolver.cached_reachability(url) if resolver is not None else None
            # Unclassified (no resolver / resolution not run) keeps the
            # conversion-only behavior: hand the URL to the provider.
            if reachability is None or reachability == UrlReachability.PUBLIC:
                return None
        if resolver is not None:
            resolved: bytes | str | None
            if self.prefers_raw_file_bytes:
                resolved = resolver.cached_bytes(url)
            else:
                resolved = resolver.cached_data_url(url)
            if resolved is not None:
                return resolved
        raise ValueError(
            f"Unresolved file URL for this provider: {url!r} (run resolution via "
            "prepare_llm_invocation_params with a file resolver set)"
        )

    def decoded_file_bytes(self, file_data: Mapping[str, Any]) -> bytes:
        """Return a ``file_base64`` item's decoded bytes, from the resolver's cache when prepared.

        Falls back to decoding inline for conversion-only calls where no
        resolution has run.

        Args:
            file_data: The item's ``file`` mapping (``file_data``, ...).
        """
        data_url = file_data["file_data"]
        if self._file_resolver is not None:
            cached = self._file_resolver.cached_bytes(data_url)
            if cached is not None:
                return cached
        return base64.b64decode(data_url.split(",", 1)[1])

    def create_llm_specific_message(self, message: Any) -> LLMSpecificMessage:
        """Create an LLM-specific message (as opposed to a standard message) for use in an LLMContext.

        Args:
            message: The message content.

        Returns:
            A LLMSpecificMessage instance.
        """
        return LLMSpecificMessage(llm=self.id_for_llm_specific_messages, message=message)

    def get_messages(
        self, context: LLMContext, *, truncate_large_values: bool = False
    ) -> list[LLMContextMessage]:
        """Get messages from the LLM context, including standard and LLM-specific messages.

        Args:
            context: The LLM context containing messages.
            truncate_large_values: If True, return deep copies of messages with
                large values replaced by short placeholders.

        Returns:
            List of messages including standard and LLM-specific messages.
        """
        return context.get_messages(
            self.id_for_llm_specific_messages, truncate_large_values=truncate_large_values
        )

    def from_standard_tools(self, tools: Any) -> list[Any] | NotGiven | None:
        """Convert tools from standard format to provider format.

        Built-in tools are automatically merged into the schema before conversion so that every
        inference request receives them without the user having to declare them explicitly.

        Args:
            tools: Tools in standard format or provider-specific format.

        Returns:
            List of tools converted to provider format, or original tools
            (possibly ``None``) if not in standard format.
        """
        if self._builtin_tools:
            if isinstance(tools, ToolsSchema):
                tools = ToolsSchema(
                    standard_tools=tools.standard_tools + list(self._builtin_tools.values()),
                    custom_tools=tools.custom_tools,
                )
            else:
                # User supplied tools in a legacy/provider-specific format.
                # Built-in tools cannot be safely merged, so they will not be injected.
                # Migrate to ToolsSchema to enable built-in tool support; use custom_tools
                # as an escape hatch for any provider-specific tools that don't fit the
                # standard schema.
                if tools is not None:
                    warnings.warn(
                        "Built-in tools (e.g. async tool cancellation) could not be injected "
                        "because the supplied tools are not a ToolsSchema instance. "
                        "Migrate to ToolsSchema to enable built-in tool support. "
                        "Use ToolsSchema(custom_tools=...) as an escape hatch for any "
                        "provider-specific tools that don't fit the standard schema.",
                        UserWarning,
                        stacklevel=2,
                    )
                # Fall through and return the original tools unchanged.

        if isinstance(tools, ToolsSchema):
            return self.to_provider_tools_format(tools)
        # Fallback to return the same tools in case they are not in a standard format
        return tools

    def _warn_context_system_message(self):
        """Warn once that the initial ``"system"`` context message is deprecated.

        The system prompt belongs on the LLM service, where it composes with
        the instructions the framework contributes — appended instructions,
        turn-completion guidance, async-tool guidance. A prompt carried in the
        context bypasses that composition, and providers that take the system
        instruction as a separate parameter drop it entirely when the service
        also has one.
        """
        if self._warned_context_system_message:
            return
        self._warned_context_system_message = True
        # Raised under an `always` filter so it survives the default
        # `ignore::DeprecationWarning` that hides call sites outside
        # `__main__` — every caller here is inside an LLM service. The flag
        # above supplies the deduplication that filter would provide.
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                'Passing the system prompt as an initial "system" message in `LLMContext` is'
                " deprecated since 1.9.0 and will be removed in 2.0.0. Set `system_instruction`"
                " on the LLM service instead.",
                DeprecationWarning,
                stacklevel=3,
            )

    def _extract_initial_system(
        self,
        messages: list,
        *,
        system_instruction: str | None = None,
    ) -> str | None:
        """Extract an initial ``"system"`` message for use as a system instruction.

        Only useful for services that expect the system instruction as a
        separate parameter, not inline in conversation history (today, all
        non-OpenAI services). Does not extract ``"developer"`` messages —
        those are converted to ``"user"`` by the adapter's subsequent message
        loop, like any other non-system role the provider doesn't support.

        Checks ``messages[0]``. If the role is ``"system"``, pops and returns
        its content. If extracting would leave the messages list empty
        (``len(messages) == 1``), the message is converted to ``"user"``
        role instead of being extracted, to prevent sending an empty
        conversation history to providers that require at least one
        non-system message.

        Args:
            messages: Message list in standard format. The list is mutated
                in-place; the message dicts it holds are never mutated, since
                they are shared with the source LLMContext.
            system_instruction: The system instruction from service settings
                or ``run_inference``. Only used to decide whether to warn
                about a conflict in the single-message case.

        Returns:
            The extracted system message content, or ``None`` if nothing
            was extracted.
        """
        if not messages:
            return None

        if messages[0].get("role") != "system":
            return None

        self._warn_context_system_message()

        # Would extracting empty the list? Convert to "user" instead.
        if len(messages) == 1:
            if system_instruction:
                if not self._warned_system_instruction:
                    self._warned_system_instruction = True
                    logger.warning(
                        "Both system_instruction and an initial system message in"
                        " context are set. Using system_instruction. The context"
                        " system message is being converted to a user message to"
                        " avoid sending an empty conversation history."
                    )
            # Replace rather than mutate: the message dicts are shared with the
            # source LLMContext, so an in-place write would rewrite its history.
            messages[0] = {**messages[0], "role": "user"}
            return None

        # Extract
        content = messages[0].get("content", "")
        if isinstance(content, list):
            # Join text parts for providers that expect a string system instruction
            content = " ".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        messages.pop(0)
        return content

    def _resolve_system_instruction(
        self,
        system_from_context: str | None,
        system_instruction: str | None,
        *,
        discard_context_system: bool,
    ) -> str | None:
        """Resolve conflict between ``system_instruction`` and an extracted context system message.

        Args:
            system_from_context: Content extracted from an initial ``"system"``
                message by :meth:`_extract_initial_system`, or detected
                inline (OpenAI adapters).
            system_instruction: From service settings or ``run_inference`` param.
            discard_context_system: If ``True`` (non-OpenAI adapters), the
                context system message is discarded when ``system_instruction``
                is also present. If ``False`` (OpenAI adapters), both are kept.

        Returns:
            The effective system instruction to use, or ``None`` if the system
            instruction is already represented in the messages (OpenAI path).
        """
        if system_from_context and system_instruction:
            if not self._warned_system_instruction:
                self._warned_system_instruction = True
                if discard_context_system:
                    # This provider takes the system instruction as a separate
                    # parameter, so only one of the two can be sent.
                    logger.warning(
                        "Both system_instruction and an initial system message in"
                        " context are set. Using system_instruction; the context"
                        " system message is not sent to the model. Move the prompt"
                        " to system_instruction on the LLM service."
                    )
                else:
                    logger.warning(
                        "Both system_instruction and an initial system message"
                        " in context are set, which may be unintended. Keeping"
                        " both, but consider using system_instruction for"
                        " system-level instructions and developer messages in"
                        " context for supplementary guidance."
                    )

        if system_instruction:
            return system_instruction

        if system_from_context:
            if discard_context_system:
                return system_from_context
            else:
                # Content is already in messages; nothing to prepend
                return None

        return None
