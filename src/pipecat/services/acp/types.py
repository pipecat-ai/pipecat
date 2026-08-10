#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Wire types for the Agent Client Protocol.

These models mirror the JSON shapes defined by the Agent Client Protocol
(https://agentclientprotocol.com). The protocol uses camelCase on the wire, so
every model here declares a camelCase alias generator and accepts either
spelling on input.

Unknown fields are preserved rather than dropped: agents may send extension
fields (and ACP reserves ``_meta`` plus underscore-prefixed members for exactly
that), and a client that discards them cannot pass them on to a renderer.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

PROTOCOL_VERSION = 1
"""The ACP major version this client speaks."""


class ACPModel(BaseModel):
    """Base for every ACP wire model: camelCase aliases, extras preserved."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="allow",
    )

    def to_wire(self) -> dict[str, Any]:
        """Serialize to a JSON-RPC-ready dict.

        Returns:
            The model as camelCase keys with unset fields omitted.
        """
        return self.model_dump(by_alias=True, exclude_none=True)


#
# Content
#


class ContentBlock(ACPModel):
    """A single piece of content in a prompt or an agent message.

    ACP reuses MCP's content block shapes. Which fields are populated depends on
    ``type``: ``text`` carries ``text``, ``image``/``audio`` carry ``data`` and
    ``mime_type``, ``resource_link`` carries ``uri``, and ``resource`` carries an
    embedded ``resource`` object.

    Parameters:
        type: The block discriminator.
        text: Text content, for ``text`` blocks.
        data: Base64 payload, for ``image`` and ``audio`` blocks.
        mime_type: Media type of ``data``.
        uri: Target, for ``resource_link`` blocks.
        name: Human-readable label, for ``resource_link`` blocks.
        resource: Embedded resource contents, for ``resource`` blocks.
    """

    type: str
    text: str | None = None
    data: str | None = None
    mime_type: str | None = None
    uri: str | None = None
    name: str | None = None
    resource: dict[str, Any] | None = None


def text_block(text: str) -> ContentBlock:
    """Build a plain text content block.

    Args:
        text: The text to wrap.

    Returns:
        A ``text`` content block.
    """
    return ContentBlock(type="text", text=text)


#
# Capabilities and initialization
#


class FileSystemCapability(ACPModel):
    """Filesystem methods the client is willing to serve.

    Parameters:
        read_text_file: Whether the client serves ``fs/read_text_file``.
        write_text_file: Whether the client serves ``fs/write_text_file``.
    """

    read_text_file: bool = False
    write_text_file: bool = False


class ClientCapabilities(ACPModel):
    """What the client can do on the agent's behalf.

    An agent that sees a capability turned off falls back to doing the work
    itself, so declaring nothing is always safe.

    Parameters:
        fs: Filesystem methods the client serves.
        terminal: Whether the client serves the ``terminal/*`` methods.
    """

    fs: FileSystemCapability = Field(default_factory=FileSystemCapability)
    terminal: bool = False


class PromptCapabilities(ACPModel):
    """Content block types the agent accepts in a prompt.

    Parameters:
        image: Whether image blocks are accepted.
        audio: Whether audio blocks are accepted.
        embedded_context: Whether embedded resource blocks are accepted.
    """

    image: bool = False
    audio: bool = False
    embedded_context: bool = False


class AgentCapabilities(ACPModel):
    """What the agent supports.

    Parameters:
        load_session: Whether ``session/load`` is supported.
        prompt_capabilities: Content block types accepted in a prompt.
    """

    load_session: bool = False
    prompt_capabilities: PromptCapabilities = Field(default_factory=PromptCapabilities)


class AuthMethod(ACPModel):
    """An authentication method the agent offers.

    Parameters:
        id: Identifier to pass to ``authenticate``.
        name: Human-readable label.
        description: Longer explanation of the method.
    """

    id: str
    name: str | None = None
    description: str | None = None


class InitializeResult(ACPModel):
    """The agent's response to ``initialize``.

    Parameters:
        protocol_version: Major protocol version the agent settled on.
        agent_capabilities: What the agent supports.
        auth_methods: Authentication methods the agent offers.
    """

    protocol_version: int = PROTOCOL_VERSION
    agent_capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    auth_methods: list[AuthMethod] = Field(default_factory=list)


class MCPServer(ACPModel):
    """An MCP server the agent should connect to for this session.

    Parameters:
        name: Label for the server.
        command: Executable to run.
        args: Arguments passed to the executable.
        env: Environment variables for the server process.
    """

    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: list[dict[str, str]] = Field(default_factory=list)


#
# Session state
#


class SessionMode(ACPModel):
    """An operating mode the agent offers.

    Parameters:
        id: Identifier to pass to ``session/set_mode``.
        name: Human-readable label.
        description: Longer explanation of the mode.
    """

    id: str
    name: str | None = None
    description: str | None = None


class SessionModeState(ACPModel):
    """The agent's current and available modes.

    Parameters:
        current_mode_id: The mode the agent is operating in.
        available_modes: Every mode the agent offers.
    """

    current_mode_id: str | None = None
    available_modes: list[SessionMode] = Field(default_factory=list)


class AvailableCommand(ACPModel):
    """A slash command the agent exposes.

    Parameters:
        name: Command name, without a leading slash.
        description: What the command does.
        input: Hint describing the command's expected input.
    """

    name: str
    description: str | None = None
    input: dict[str, Any] | None = None


class NewSessionResult(ACPModel):
    """The agent's response to ``session/new``.

    Parameters:
        session_id: Identifier for the new session.
        modes: The session's initial mode state, if the agent offers modes.
    """

    session_id: str
    modes: SessionModeState | None = None


#
# Plans and tool calls
#


class PlanEntryStatus(StrEnum):
    """Status of a single plan entry."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class PlanEntry(ACPModel):
    """One step in the agent's plan.

    Parameters:
        content: What the step does.
        priority: Relative importance, as reported by the agent.
        status: Where the step is in its lifecycle.
    """

    content: str
    priority: str | None = None
    status: PlanEntryStatus = PlanEntryStatus.PENDING


class ToolKind(StrEnum):
    """Broad category of a tool call, used to pick an icon or a phrasing."""

    READ = "read"
    EDIT = "edit"
    DELETE = "delete"
    MOVE = "move"
    SEARCH = "search"
    EXECUTE = "execute"
    THINK = "think"
    FETCH = "fetch"
    SWITCH_MODE = "switch_mode"
    OTHER = "other"


class ToolCallStatus(StrEnum):
    """Where a tool call is in its lifecycle."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolCallLocation(ACPModel):
    """A file the tool call touches.

    Parameters:
        path: Absolute path to the file.
        line: Line number of interest within the file.
    """

    path: str
    line: int | None = None


class ToolCall(ACPModel):
    """A tool call the agent has started.

    Parameters:
        tool_call_id: Identifier used by every later update.
        title: Human-readable description of the call.
        kind: Broad category of the call.
        status: Where the call is in its lifecycle.
        content: Output produced by the call, including diffs.
        locations: Files the call touches.
        raw_input: The arguments the agent passed to the tool.
        raw_output: The raw result the tool returned.
    """

    tool_call_id: str
    title: str | None = None
    kind: ToolKind = ToolKind.OTHER
    status: ToolCallStatus = ToolCallStatus.PENDING
    content: list[dict[str, Any]] = Field(default_factory=list)
    locations: list[ToolCallLocation] = Field(default_factory=list)
    raw_input: Any | None = None
    raw_output: Any | None = None


class ToolCallUpdate(ACPModel):
    """A change to a tool call already announced.

    Every field other than ``tool_call_id`` is a delta: absent means unchanged,
    so a consumer needs the original :class:`ToolCall` to render an update in
    full.

    Parameters:
        tool_call_id: Identifier of the call being updated.
        title: Replacement description, if it changed.
        kind: Replacement category, if it changed.
        status: New lifecycle status, if it changed.
        content: Output produced since the last update.
        locations: Replacement file list, if it changed.
        raw_input: Replacement arguments, if they changed.
        raw_output: The raw result the tool returned.
    """

    tool_call_id: str
    title: str | None = None
    kind: ToolKind | None = None
    status: ToolCallStatus | None = None
    content: list[dict[str, Any]] | None = None
    locations: list[ToolCallLocation] | None = None
    raw_input: Any | None = None
    raw_output: Any | None = None


#
# Permission
#


class PermissionOptionKind(StrEnum):
    """The effect of choosing a permission option."""

    ALLOW_ONCE = "allow_once"
    ALLOW_ALWAYS = "allow_always"
    REJECT_ONCE = "reject_once"
    REJECT_ALWAYS = "reject_always"


class PermissionOption(ACPModel):
    """One choice offered in a permission request.

    Parameters:
        option_id: Identifier to send back in the response.
        name: Human-readable label for the choice.
        kind: The effect of choosing this option.
    """

    option_id: str
    name: str | None = None
    kind: PermissionOptionKind = PermissionOptionKind.ALLOW_ONCE


class RequestPermissionParams(ACPModel):
    """Parameters of a ``session/request_permission`` call.

    Parameters:
        session_id: Session the request belongs to.
        tool_call: The call awaiting approval.
        options: The choices the user may pick from.
    """

    session_id: str
    tool_call: ToolCall
    options: list[PermissionOption] = Field(default_factory=list)


class RequestPermissionResult(ACPModel):
    """The client's answer to a permission request.

    Parameters:
        outcome: Either ``{"outcome": "selected", "optionId": ...}`` or
            ``{"outcome": "cancelled"}``.
    """

    outcome: dict[str, Any]

    @classmethod
    def selected(cls, option_id: str) -> "RequestPermissionResult":
        """Build a result selecting an option.

        Args:
            option_id: The chosen option's identifier.

        Returns:
            A populated result.
        """
        return cls(outcome={"outcome": "selected", "optionId": option_id})

    @classmethod
    def cancelled(cls) -> "RequestPermissionResult":
        """Build a result declining to answer.

        Returns:
            A populated result.
        """
        return cls(outcome={"outcome": "cancelled"})


#
# Filesystem and terminal
#


class ReadTextFileParams(ACPModel):
    """Parameters of an ``fs/read_text_file`` call.

    Parameters:
        session_id: Session the request belongs to.
        path: Absolute path to read.
        line: First line to return, 1-indexed.
        limit: Maximum number of lines to return.
    """

    session_id: str
    path: str
    line: int | None = None
    limit: int | None = None


class ReadTextFileResult(ACPModel):
    """The contents returned for an ``fs/read_text_file`` call.

    Parameters:
        content: The requested text.
    """

    content: str


class WriteTextFileParams(ACPModel):
    """Parameters of an ``fs/write_text_file`` call.

    Parameters:
        session_id: Session the request belongs to.
        path: Absolute path to write.
        content: The full new contents of the file.
    """

    session_id: str
    path: str
    content: str


class TerminalParams(ACPModel):
    """Parameters of any ``terminal/*`` call.

    The protocol splits these across several methods with overlapping shapes;
    this model covers all of them, leaving unused fields unset.

    Parameters:
        session_id: Session the request belongs to.
        terminal_id: Identifier of an existing terminal.
        command: Executable to run, for ``terminal/create``.
        args: Arguments passed to the executable.
        env: Environment variables for the process.
        cwd: Working directory for the process.
        output_byte_limit: Maximum bytes of output to retain.
    """

    session_id: str
    terminal_id: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: list[dict[str, str]] = Field(default_factory=list)
    cwd: str | None = None
    output_byte_limit: int | None = None


#
# Turn completion
#


class StopReason(StrEnum):
    """Why a ``session/prompt`` turn ended."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    MAX_TURN_REQUESTS = "max_turn_requests"
    REFUSAL = "refusal"
    CANCELLED = "cancelled"


class ACPErrorData(ACPModel):
    """A JSON-RPC error returned to or by this client.

    Parameters:
        code: JSON-RPC error code.
        message: Human-readable description.
        data: Additional error detail.
    """

    code: int = -32603
    message: str = "Internal error"
    data: Any | None = None
