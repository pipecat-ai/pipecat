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
config is bound to the application's tools.

The config loads from YAML, JSON, or a plain dict, and is validated
structurally on load: the initial node exists, every transition names a node,
tool names are unique within a node, and every action is well-formed. Binding
validates the references to code.

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
            transition_to: pizza
          - name: choose_sushi
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
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipecat.flows.types import ContextStrategy
from pipecat.utils.yaml import include_loader


class FlowConfig(BaseModel):
    """A conversation flow described as data.

    Parameters:
        initial_node: Name of the node the flow starts in.
        nodes: The flow's nodes, keyed by name.
        global_functions: Tools offered at every node.
    """

    model_config = ConfigDict(extra="forbid")

    class Message(BaseModel):
        """One message in a node's ``task_messages``.

        Parameters:
            role: Message role, e.g. ``developer`` or ``system``.
            content: Message text. May contain ``{{ variable }}`` placeholders
                substituted at bind time.
        """

        model_config = ConfigDict(extra="forbid")

        role: str
        content: str

    class Branch(BaseModel):
        """A transition chosen by a field of the tool's result.

        Parameters:
            field: Key of the tool's result whose value selects the case.
            cases: Result value to node name.
            default: Node to transition to when the value matches no case.
                When omitted, an unmatched value stays on the current node.
        """

        model_config = ConfigDict(extra="forbid")

        field: str
        cases: dict[str, str] = Field(min_length=1)
        default: str | None = None

        def targets(self) -> list[str]:
            """Every node name this branch can transition to."""
            return list(self.cases.values()) + ([self.default] if self.default else [])

    class Function(BaseModel):
        """A tool offered at a node, referenced by name.

        Parameters:
            name: Name of a Flows direct function in the bound tools. The
                tool's description and parameters come from that function.
            transition_to: Node to transition to after the tool completes,
                or a :class:`FlowConfig.Branch`. Omitted for tools that stay
                on the current node.
        """

        model_config = ConfigDict(extra="forbid")

        name: str
        transition_to: "str | FlowConfig.Branch | None" = None

        def targets(self) -> list[str]:
            """Every node name this function can transition to."""
            if self.transition_to is None:
                return []
            if isinstance(self.transition_to, str):
                return [self.transition_to]
            return self.transition_to.targets()

    class Action(BaseModel):
        """A pre- or post-action on a node.

        Built-in action types (``tts_say``, ``end_conversation``) need nothing
        else. The ``function`` type names a handler in the bound tools. Custom
        types registered with ``FlowManager.register_action`` are referenced by
        type alone. Any additional keys pass through to the action handler.

        Parameters:
            type: Action type identifier.
            handler: For the ``function`` type, the name of the handler.
        """

        model_config = ConfigDict(extra="allow")

        type: str
        handler: str | None = None

        @model_validator(mode="after")
        def _check_handler(self) -> "FlowConfig.Action":
            if self.type == "function" and not self.handler:
                raise ValueError("a 'function' action requires a 'handler' name")
            if self.type != "function" and self.handler is not None:
                raise ValueError(
                    f"action type '{self.type}' does not take a 'handler'; "
                    "register custom action types with FlowManager.register_action"
                )
            return self

        def extras(self) -> dict[str, Any]:
            """The pass-through keys beyond ``type`` and ``handler``."""
            return dict(self.model_extra or {})

    class Node(BaseModel):
        """One node of the flow.

        Parameters:
            task_messages: What the LLM should do at this node.
            role_message: The bot's role or personality, sent as the LLM's
                system instruction on entering this node. It persists across
                transitions until another node sets its own.
            functions: Tools offered at this node, in addition to the
                config's ``global_functions``.
            pre_actions: Actions run before the LLM responds at this node.
            post_actions: Actions run after the LLM responds at this node.
            context_strategy: How the LLM context is updated on entering this
                node. Defaults to the ``FlowManager``'s strategy.
            respond_immediately: Whether the LLM responds as soon as the node
                is entered. Defaults to True.
        """

        model_config = ConfigDict(extra="forbid")

        task_messages: "list[FlowConfig.Message]"
        role_message: str | None = None
        functions: "list[FlowConfig.Function]" = Field(default_factory=list)
        pre_actions: "list[FlowConfig.Action]" = Field(default_factory=list)
        post_actions: "list[FlowConfig.Action]" = Field(default_factory=list)
        context_strategy: Literal["append", "reset"] | None = None
        respond_immediately: bool = True

        @model_validator(mode="after")
        def _check_unique_function_names(self) -> "FlowConfig.Node":
            _check_unique([f.name for f in self.functions], "node")
            return self

        def context_strategy_enum(self) -> ContextStrategy | None:
            """The node's ``context_strategy`` as a :class:`ContextStrategy`."""
            return ContextStrategy(self.context_strategy) if self.context_strategy else None

    initial_node: str
    nodes: dict[str, Node] = Field(min_length=1)
    global_functions: list[Function] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_graph(self) -> "FlowConfig":
        if self.initial_node not in self.nodes:
            raise ValueError(f"initial_node '{self.initial_node}' is not a defined node")

        _check_unique([f.name for f in self.global_functions], "global_functions")
        global_names = {f.name for f in self.global_functions}

        for node_name, node in self.nodes.items():
            for func in node.functions:
                if func.name in global_names:
                    raise ValueError(
                        f"node '{node_name}' function '{func.name}' is also a global function"
                    )
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
            base_dir: Directory that ``!include`` paths resolve against. When
                omitted, ``!include`` is unavailable.

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

        YAML files may use ``!include`` with paths relative to the file's
        directory.

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


def _check_targets(func: FlowConfig.Function, nodes: dict[str, Any], where: str) -> None:
    for target in func.targets():
        if target not in nodes:
            raise ValueError(
                f"{where} function '{func.name}' transitions to unknown node '{target}'"
            )
