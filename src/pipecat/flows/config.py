#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Declarative flow configuration.

A :class:`FlowConfig` describes a conversation flow as data: the nodes, what
each one says, which tools each node offers, and where each tool leads. It
contains no Python callables. Every tool a node references is a Flows direct
function that lives in the application's code and is resolved by name when the
config is joined to the application's handlers by constructing a
:class:`~pipecat.flows.Flow`. The exception is a ``transition_only`` function,
which does nothing but move the conversation to another node: the config
supplies its description and it needs no code.

The config loads from YAML, JSON, or a plain dict, and is validated
structurally on load: the initial node exists, every transition names a node,
tool names are unique within a node, and every action is well-formed.
Constructing a :class:`~pipecat.flows.Flow` validates the references to code.

Example YAML::

    initial_node: greet

    nodes:
      greet:
        role_message: You are a friendly order-taking assistant.
        task_messages:
          - role: developer
            content: Greet the caller and ask whether they want pizza or sushi.
        functions:
          - name: choose_pizza
            transition_only: true
            description: The caller wants to order pizza.
            transition_to: pizza
          - name: choose_sushi
            transition_only: true
            description: The caller wants to order sushi.
            transition_to: sushi

      pizza:
        task_messages:
          - role: developer
            content: Take a pizza order.
        functions:
          - name: select_pizza_order
            transition_to:
              field: status
              cases:
                ok: confirm
                unavailable: pizza
              default: confirm

    global_functions:
      - name: get_delivery_estimate
"""

import json
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from pipecat.flows.types import ContextStrategy
from pipecat.utils.yaml import include_loader

BUILT_IN_ACTIONS_WITHOUT_HANDLER = frozenset({"tts_say", "end_conversation"})
"""Built-in action types whose behavior is fixed, so a ``handler`` is not allowed."""

BUILT_IN_ACTIONS = BUILT_IN_ACTIONS_WITHOUT_HANDLER | {"function"}
"""Every action type the runtime provides without registration."""

RETURN = "return"
"""The transition target that, inside a subflow, leads to the ``return_to`` of the node running it."""


def case_key(value: Any) -> str:
    """The canonical string a branch matches a case key or result value on.

    Booleans, and strings spelling one in any case, become ``true`` and
    ``false``; everything else is its ``str()``. So ``true:``, ``"True":``,
    and a result of Python ``True`` all meet at the same case.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    lowered = text.lower()
    return lowered if lowered in ("true", "false") else text


class FlowConfig(BaseModel):
    r"""A conversation flow described as data.

    Load one with :meth:`from_file` for a YAML or JSON file, :meth:`from_yaml`
    for YAML text, :meth:`from_json` for JSON text, or Pydantic's
    ``model_validate`` for a dict that is already parsed.

    ``!include`` resolves paths on the application's file system and can read
    any file the process can read, so it belongs only in configs the
    application ships. Load a config from another source, such as a request
    body, with :meth:`from_yaml` and no ``base_dir``, which rejects
    ``!include``, or with :meth:`from_json`.

    Prompt text may refer to the manager's state with ``{{ key }}``
    placeholders: a node's ``role_message``, the ``content`` of its
    ``task_messages``, and the ``text`` of a ``tts_say`` action.
    :class:`~pipecat.flows.FlowManager` fills them from ``flow_manager.state``
    each time it enters the node, so a value stored by a handler earlier in the
    conversation can appear in a later prompt. ``{{ order.size }}`` walks into
    a stored mapping, and values are rendered with ``str()``. A key that is not
    in state raises :class:`~pipecat.flows.FlowError` when the node is entered.
    To show the LLM a literal ``{{ key }}``, escape it as ``\{{ key }}``.

    Metadata: a node and a function each take an optional ``metadata``
    mapping of JSON values, for whatever the application wants to keep with
    them, such as the id of the element a node stands for in an editor. Flows
    carries it and never acts on it, and it is never sent to the LLM. A
    :class:`~pipecat.flows.Flow` copies it onto the node configs and function
    schemas it builds, and :class:`~pipecat.flows.FlowManager` passes it to
    ``on_node_entered`` handlers and to an RTVI client as the flow moves.
    Metadata is sent to the client, so keep anything the end user should not
    see out of it.

    Subflows: a set of nodes used by several flows, or several times in one,
    is written once as a subflow under ``flows``. A node runs it by naming it
    as its ``flow``, in place of prompts and functions of its own, and says
    where the subflow's ``return`` leads::

        initial_node: verify_caller
        flows:
          verification: !include verification.yaml
        nodes:
          verify_caller:
            flow: verification
            return_to: book
          book:
            ...

    Entering such a node enters the subflow at its first node. Inside a
    subflow a transition target names one of its own nodes, or is ``return``,
    which leads to the ``return_to`` of the node running it; to return to two
    places, run the subflow from two nodes. ``return_to`` may be omitted for a
    subflow that never returns. A :class:`~pipecat.flows.Flow` builds a
    subflow's nodes once for each node that runs it, named
    ``<node>.<subflow node>``, so everything before the dot is the node that
    runs the subflow; that is the name ``FlowManager.current_node`` reports,
    and ``on_node_entered`` gives the running node separately as ``parent``. A
    config that has subflows therefore cannot have a dot in a node name. A
    subflow's nodes cannot run subflows, and ``global_functions`` are offered
    at them too.

    Parameters:
        initial_node: Name of the node the flow starts in.
        nodes: The flow's nodes, keyed by name.
        global_functions: Tools offered at every node.
        flows: Subflows the flow's nodes can run, keyed by flow id.
    """

    model_config = ConfigDict(extra="forbid")

    class Message(BaseModel):
        """One message in a node's ``task_messages``.

        Parameters:
            role: Message role, e.g. ``developer`` or ``system``.
            content: Message text. May contain ``{{ key }}`` placeholders;
                see :class:`FlowConfig`.
        """

        model_config = ConfigDict(extra="forbid")

        role: str
        content: str

    class Branch(BaseModel):
        """A transition chosen by a field of the tool's result.

        Parameters:
            field: Key of the tool's result whose value selects the case.
            cases: Result value to node name. Keys may be written as strings,
                booleans, or numbers; they match the result value by its
                canonical string (see :func:`case_key`), so ``true:`` matches a
                result of ``True``.
            default: Node to transition to when the value matches no case.
                When omitted, an unmatched value stays on the current node.
        """

        model_config = ConfigDict(extra="forbid")

        field: str
        cases: dict[str, str] = Field(min_length=1)
        default: str | None = None

        @field_validator("cases", mode="before")
        @classmethod
        def _canonical_keys(cls, value: Any) -> Any:
            if isinstance(value, Mapping):
                return {case_key(k): v for k, v in value.items()}
            return value

        def targets(self) -> list[str]:
            """Every node name this branch can transition to."""
            return list(self.cases.values()) + ([self.default] if self.default else [])

    class Function(BaseModel):
        """A tool offered at a node.

        Ordinarily the entry names a Flows direct function in the handlers a
        :class:`~pipecat.flows.Flow` is constructed with, and the tool's
        description and parameters come from that function. A
        ``transition_only`` entry is defined entirely here instead: it takes
        no parameters, runs no code, and moves the conversation to
        ``transition_to`` when the LLM calls it.

        Parameters:
            name: The tool's name, as the LLM sees it. For an ordinary
                entry, also the name of the direct function in the handlers.
            transition_only: Whether the tool is defined here rather than in
                code. Requires ``description`` and a ``transition_to`` that
                names a node.
            description: What the tool is for, for the LLM. Only a
                ``transition_only`` entry has one; a direct function
                describes itself in its docstring.
            transition_to: Node to transition to after the tool completes,
                or a :class:`FlowConfig.Branch`. Omitted for tools that stay
                on the current node. Inside a subflow a target may also be
                ``return``; see "Subflows" in :class:`FlowConfig`.
            metadata: JSON data the application attaches to the tool. See
                "Metadata" in :class:`FlowConfig`.
        """

        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={
                "if": {
                    "properties": {"transition_only": {"const": True}},
                    "required": ["transition_only"],
                },
                "then": {
                    "required": ["description", "transition_to"],
                    "properties": {"transition_to": {"type": "string"}},
                },
                "else": {"not": {"required": ["description"]}},
            },
        )

        name: str
        transition_only: bool = False
        description: str | None = None
        transition_to: "str | FlowConfig.Branch | None" = None
        metadata: dict[str, JsonValue] | None = None

        @model_validator(mode="after")
        def _check_shape(self) -> "FlowConfig.Function":
            if self.transition_only:
                if not self.description:
                    raise ValueError(
                        f"function '{self.name}' is transition_only and needs a description"
                    )
                if not isinstance(self.transition_to, str):
                    raise ValueError(
                        f"function '{self.name}' is transition_only and must name the node "
                        "it transitions to"
                    )
            elif self.description is not None:
                raise ValueError(
                    f"function '{self.name}' has a description, which only a transition_only "
                    "function takes; a direct function describes itself in its docstring"
                )
            return self

        def targets(self) -> list[str]:
            """Every node name this function can transition to."""
            if self.transition_to is None:
                return []
            if isinstance(self.transition_to, str):
                return [self.transition_to]
            return self.transition_to.targets()

    class Action(BaseModel):
        """A pre- or post-action on a node.

        The built-in ``tts_say`` and ``end_conversation`` types take no
        handler. The built-in ``function`` type requires one: the handler runs
        inline in the pipeline, queued behind the bot's turn. A custom type
        may name a handler too, which then runs immediately when the node's
        actions execute; a custom type without one must be registered in code
        with ``FlowManager.register_action``. Any additional keys pass through
        to the handler. The ``text`` of a ``tts_say`` action may contain
        ``{{ key }}`` placeholders; see :class:`FlowConfig`.

        Parameters:
            type: Action type identifier.
            handler: Name of the handler in the handlers a
                :class:`~pipecat.flows.Flow` is constructed with. Required
                for ``function``, optional for custom types, not allowed on
                ``tts_say`` or ``end_conversation``.
        """

        model_config = ConfigDict(extra="allow")

        type: str
        handler: str | None = None

        @model_validator(mode="after")
        def _check_handler(self) -> "FlowConfig.Action":
            if self.type == "function" and not self.handler:
                raise ValueError("a 'function' action requires a 'handler' name")
            if self.type in BUILT_IN_ACTIONS_WITHOUT_HANDLER and self.handler is not None:
                raise ValueError(f"the built-in '{self.type}' action does not take a 'handler'")
            return self

        @property
        def registered_in_code(self) -> bool:
            """Whether this is a custom type whose handler the config does not name."""
            return self.type not in BUILT_IN_ACTIONS and self.handler is None

        def extras(self) -> dict[str, Any]:
            """The pass-through keys beyond ``type`` and ``handler``."""
            return dict(self.model_extra or {})

    class Node(BaseModel):
        """One node of the flow.

        A node either prompts the LLM, with ``task_messages`` and the fields
        that go with them, or runs a subflow, with ``flow`` and ``return_to``
        in their place. See "Subflows" in :class:`FlowConfig`.

        Parameters:
            task_messages: What the LLM should do at this node. Required
                unless the node runs a subflow.
            role_message: The bot's role or personality, sent as the LLM's
                system instruction on entering this node. It persists across
                transitions until another node sets its own. May contain
                ``{{ key }}`` placeholders; see :class:`FlowConfig`.
            functions: Tools offered at this node, in addition to the
                config's ``global_functions``.
            pre_actions: Actions run before the LLM responds at this node.
            post_actions: Actions run after the LLM responds at this node.
            context_strategy: How the LLM context is updated on entering this
                node. Defaults to the ``FlowManager``'s strategy.
            respond_immediately: Whether the LLM responds as soon as the node
                is entered. Defaults to True.
            metadata: JSON data the application attaches to the node. See
                "Metadata" in :class:`FlowConfig`.
            flow: Id of the subflow this node runs, a key of the config's
                ``flows``. A node that runs a subflow takes only
                ``return_to`` and ``metadata`` besides.
            return_to: The node that ``return`` in the subflow leads to. May
                be omitted when the subflow never returns.
        """

        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={
                "if": {"properties": {"flow": {"type": "string"}}, "required": ["flow"]},
                "then": {
                    "properties": {
                        "task_messages": {"maxItems": 0},
                        "role_message": {"type": "null"},
                        "functions": {"maxItems": 0},
                        "pre_actions": {"maxItems": 0},
                        "post_actions": {"maxItems": 0},
                        "context_strategy": {"type": "null"},
                        "respond_immediately": {"const": True},
                    }
                },
                "else": {
                    "required": ["task_messages"],
                    "properties": {"return_to": {"type": "null"}},
                },
            },
        )

        task_messages: "list[FlowConfig.Message]" = Field(default_factory=list)
        role_message: str | None = None
        functions: "list[FlowConfig.Function]" = Field(default_factory=list)
        pre_actions: "list[FlowConfig.Action]" = Field(default_factory=list)
        post_actions: "list[FlowConfig.Action]" = Field(default_factory=list)
        context_strategy: Literal["append", "reset"] | None = None
        respond_immediately: bool = True
        metadata: dict[str, JsonValue] | None = None
        flow: str | None = None
        return_to: str | None = None

        @model_validator(mode="after")
        def _check_shape(self) -> "FlowConfig.Node":
            if self.flow is None:
                if "task_messages" not in self.model_fields_set:
                    raise ValueError("a node needs 'task_messages', or a 'flow' to run")
                if self.return_to is not None:
                    raise ValueError("'return_to' applies only to a node that runs a 'flow'")
            elif (
                self.task_messages
                or self.role_message is not None
                or self.functions
                or self.pre_actions
                or self.post_actions
                or self.context_strategy is not None
                or not self.respond_immediately
            ):
                raise ValueError(
                    f"a node that runs flow '{self.flow}' takes only 'flow', 'return_to' "
                    "and 'metadata'"
                )
            _check_unique([f.name for f in self.functions], "node")
            return self

        def context_strategy_enum(self) -> ContextStrategy | None:
            """The node's ``context_strategy`` as a :class:`ContextStrategy`."""
            return ContextStrategy(self.context_strategy) if self.context_strategy else None

    class Subflow(BaseModel):
        """A set of nodes that a flow's nodes can run.

        Inside a subflow a function transitions to one of the subflow's own
        nodes, or to ``return``. See "Subflows" in :class:`FlowConfig`.

        Parameters:
            initial_node: Name of the node the subflow starts in.
            nodes: The subflow's nodes, keyed by name.
        """

        model_config = ConfigDict(extra="forbid")

        initial_node: str
        nodes: "dict[str, FlowConfig.Node]" = Field(min_length=1)

        @model_validator(mode="after")
        def _check_graph(self) -> "FlowConfig.Subflow":
            if RETURN in self.nodes:
                raise ValueError(
                    f"a subflow cannot have a node named '{RETURN}', which is how its "
                    "functions leave it"
                )
            _check_no_dots(self.nodes)
            if self.initial_node not in self.nodes:
                raise ValueError(f"initial_node '{self.initial_node}' is not a defined node")
            targets = {*self.nodes, RETURN}
            for node_name, node in self.nodes.items():
                if node.flow is not None:
                    raise ValueError(
                        f"node '{node_name}' runs a flow, which a subflow's nodes cannot do"
                    )
                for func in node.functions:
                    _check_targets(func, targets, f"node '{node_name}'")
            return self

        def returns(self) -> bool:
            """Whether any function in the subflow transitions to ``return``."""
            return any(
                RETURN in func.targets() for node in self.nodes.values() for func in node.functions
            )

    initial_node: str
    nodes: dict[str, Node] = Field(min_length=1)
    global_functions: list[Function] = Field(default_factory=list)
    flows: dict[str, Subflow] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_graph(self) -> "FlowConfig":
        if self.flows:
            # A subflow's nodes are built as ``<node>.<subflow node>``.
            _check_no_dots(self.nodes)
        if self.initial_node not in self.nodes:
            raise ValueError(f"initial_node '{self.initial_node}' is not a defined node")

        run: set[str] = set()
        for node_name, node in self.nodes.items():
            if node.flow is None:
                continue
            subflow = self.flows.get(node.flow)
            if subflow is None:
                raise ValueError(f"node '{node_name}' runs unknown flow '{node.flow}'")
            run.add(node.flow)
            if node.return_to is None:
                if subflow.returns():
                    raise ValueError(
                        f"node '{node_name}' runs flow '{node.flow}', which returns, and "
                        "needs a 'return_to'"
                    )
            elif node.return_to not in self.nodes:
                raise ValueError(f"node '{node_name}' returns to unknown node '{node.return_to}'")
        for flow_id in self.flows:
            if flow_id not in run:
                raise ValueError(f"flow '{flow_id}' is not run by any node")

        _check_unique([f.name for f in self.global_functions], "global_functions")
        global_names = {f.name for f in self.global_functions}

        scopes = [("", self.nodes)] + [
            (f"flow '{flow_id}' ", flow.nodes) for flow_id, flow in self.flows.items()
        ]
        for scope, nodes in scopes:
            for node_name, node in nodes.items():
                for func in node.functions:
                    if func.name in global_names:
                        raise ValueError(
                            f"{scope}node '{node_name}' function '{func.name}' is also a "
                            "global function"
                        )
        for node_name, node in self.nodes.items():
            for func in node.functions:
                _check_targets(func, self.nodes, f"node '{node_name}'")
        for func in self.global_functions:
            _check_targets(func, self.nodes, "global_functions")
        return self

    @classmethod
    def from_yaml(cls, text: str, *, base_dir: Path | None = None) -> "FlowConfig":
        """Load a config from YAML text.

        Args:
            text: The YAML document.
            base_dir: Directory that the document's ``!include`` paths resolve
                against; an included file's own includes resolve against its
                directory. When omitted, ``!include`` is unavailable. An
                include can read any file the process can read, so omit
                ``base_dir`` for text from an untrusted source.

        Returns:
            The validated config.
        """
        loader = include_loader(base_dir) if base_dir is not None else yaml.SafeLoader
        return cls.model_validate(_require_mapping(yaml.load(text, loader)))

    @classmethod
    def from_json(cls, text: str) -> "FlowConfig":
        """Load a config from JSON text.

        Args:
            text: The JSON document.

        Returns:
            The validated config.
        """
        return cls.model_validate(_require_mapping(json.loads(text)))

    @classmethod
    def from_file(cls, path: str | Path) -> "FlowConfig":
        """Load a config from a ``.yaml``, ``.yml``, or ``.json`` file.

        YAML files may use ``!include`` with paths relative to the directory of
        the file that names them. Included paths are not confined to that
        directory, so load only files you control. For a config from another
        source, such as a request body, use :meth:`from_yaml` without
        ``base_dir`` or :meth:`from_json`.

        Args:
            path: Path to the file.

        Returns:
            The validated config.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            return cls.from_json(text)
        return cls.from_yaml(text, base_dir=path.parent)


def _require_mapping(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("flow config: top level must be a mapping")
    return data


def _check_unique(names: list[str], where: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate function '{name}' in {where}")
        seen.add(name)


def _check_no_dots(names: Iterable[str]) -> None:
    for name in names:
        if "." in name:
            raise ValueError(
                f"node '{name}' has a '.' in its name, which a config that has subflows cannot have"
            )


def _check_targets(func: FlowConfig.Function, targets: Collection[str], where: str) -> None:
    for target in func.targets():
        if target not in targets:
            raise ValueError(
                f"{where} function '{func.name}' transitions to unknown node '{target}'"
            )
