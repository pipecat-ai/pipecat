#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A flow config joined to the application's tools.

A :class:`Flow` is a :class:`~pipecat.flows.FlowConfig` joined to the code it
names: the config's nodes turned into runnable :data:`~pipecat.flows.NodeConfig` dicts, with every tool
reference resolved to a Flows direct function, every template variable
substituted, and every transition wired to the node the config names.

Constructing a flow validates the references the config could not: each tool exists and
has a valid direct-function signature, each ``function`` action names a
callable, and each ``{{ variable }}`` has a value. Tool return values are
checked at call time against the configured-flow contract: a tool returns
``(result, None)`` and never chooses the next node itself.
"""

import inspect
import re
import typing
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

from pipecat.flows.exceptions import FlowError, FlowProblem, FlowReferenceError
from pipecat.flows.types import (
    NO_RESPONSE,
    ActionConfig,
    ConsolidatedFunctionResult,
    ContextStrategyConfig,
    FlowArgs,
    FlowsDirectFunction,
    FlowsDirectFunctionWrapper,
    FlowsFunctionSchema,
    NodeConfig,
)

if TYPE_CHECKING:
    from pipecat.flows.config import FlowConfig
    from pipecat.flows.manager import FlowManager

_VARIABLE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class Flow:
    """A :class:`~pipecat.flows.FlowConfig` joined to tools and variables.

    The flow hands ready-made node configs to
    :class:`~pipecat.flows.FlowManager`::

        config = FlowConfig.from_file("flow.yaml")
        flow = Flow(config, tools=tools, variables={"restaurant_name": "Luigi's"})
        flow_manager = FlowManager(..., global_functions=flow.global_functions)
        await flow_manager.initialize(flow.initial_node)
    """

    def __init__(
        self,
        config: "FlowConfig",
        *,
        tools: Mapping[str, Callable] | Any,
        variables: Mapping[str, Any] | None = None,
    ):
        """Join a config to the tools and variables it refers to.

        Args:
            config: The flow config.
            tools: Where tool and action-handler names resolve. A mapping of
                names to callables, or any object whose attributes are the
                callables, typically a module. Only the names the config
                references are looked up.
            variables: Values for the ``{{ variable }}`` placeholders in the
                config's messages and action text.

        Raises:
            ~pipecat.flows.FlowReferenceError: If a referenced tool or handler
                is missing, a tool is not a valid direct function, or a
                template variable has no value. Every such problem is
                collected and reported together.
        """
        self._config = config
        self._tools = tools
        self._variables = dict(variables or {})
        self._problems: list[FlowProblem] = []

        self._nodes: dict[str, NodeConfig] = {
            name: self._build_node(name, node) for name, node in config.nodes.items()
        }
        self._global_functions = [
            schema
            for ref in config.global_functions
            if (schema := self._build_function(ref, where="global_functions", node=None))
        ]
        # Building continues past each problem so all of them are found; the
        # partially built state never escapes because this raise comes last.
        if self._problems:
            raise FlowReferenceError(self._problems)

    @property
    def config(self) -> "FlowConfig":
        """The config this flow was constructed from."""
        return self._config

    @property
    def initial_node(self) -> NodeConfig:
        """The node config the flow starts in."""
        return self._nodes[self._config.initial_node]

    @property
    def global_functions(self) -> list[FlowsFunctionSchema | FlowsDirectFunction]:
        """Tools available at every node, for ``FlowManager(global_functions=...)``.

        A fresh list each time, so the caller may extend it.
        """
        return list(self._global_functions)

    def node(self, name: str) -> NodeConfig:
        """The node config for ``name``.

        Args:
            name: A node name from the config.

        Raises:
            ~pipecat.flows.FlowError: If the config has no such node.
        """
        try:
            return self._nodes[name]
        except KeyError:
            raise FlowError(f"flow has no node '{name}'") from None

    # Building

    def _build_node(self, name: str, node: "FlowConfig.Node") -> NodeConfig:
        where = f"node '{name}'"
        config: NodeConfig = {
            "name": name,
            "task_messages": [
                {"role": m.role, "content": self._render(m.content, f"{where} task_messages")}
                for m in node.task_messages
            ],
            "respond_immediately": node.respond_immediately,
        }
        if node.role_message is not None:
            config["role_message"] = self._render(node.role_message, f"{where} role_message")
        if node.functions:
            config["functions"] = [
                schema
                for ref in node.functions
                if (schema := self._build_function(ref, where=where, node=name))
            ]
        if node.pre_actions:
            config["pre_actions"] = [
                self._build_action(a, where=f"{where} pre_actions", node=name)
                for a in node.pre_actions
            ]
        if node.post_actions:
            config["post_actions"] = [
                self._build_action(a, where=f"{where} post_actions", node=name)
                for a in node.post_actions
            ]
        strategy = node.context_strategy_enum()
        if strategy is not None:
            config["context_strategy"] = ContextStrategyConfig(strategy=strategy)
        return config

    def _build_function(
        self, ref: "FlowConfig.Function", *, where: str, node: str | None
    ) -> FlowsFunctionSchema | None:
        """Build the schema for one function entry, or record why it cannot be built."""
        function = self._lookup(ref.name, kind="tool", where=where, node=node)
        if function is None:
            return None
        try:
            wrapper = FlowsDirectFunctionWrapper(function)
        except Exception as e:
            self._problem(
                "invalid_tool",
                f"{where} tool '{ref.name}' is not a valid direct function: {e}",
                node=node,
                function=ref.name,
            )
            return None
        _warn_if_annotated_with_node(function, ref.name, where)

        schema = wrapper.to_function_schema()
        return FlowsFunctionSchema(
            name=schema.name,
            description=schema.description,
            properties=schema.properties,
            required=schema.required,
            handler=self._make_handler(ref, wrapper, where),
            cancel_on_interruption=wrapper.cancel_on_interruption,
            timeout_secs=wrapper.timeout_secs,
        )

    def _build_action(self, action: "FlowConfig.Action", *, where: str, node: str) -> ActionConfig:
        config: ActionConfig = {"type": action.type}
        if action.handler is not None:
            handler = self._lookup(action.handler, kind="action handler", where=where, node=node)
            if handler is not None:
                config["handler"] = handler
        for key, value in action.extras().items():
            if key == "text" and isinstance(value, str):
                value = self._render(value, f"{where} {action.type} text")
            config[key] = value  # type: ignore[literal-required]
        return config

    def _lookup(self, name: str, *, kind: str, where: str, node: str | None) -> Callable | None:
        """Resolve ``name`` in the tools, or record why it cannot be resolved."""
        if isinstance(self._tools, Mapping):
            target = self._tools.get(name)
        else:
            target = getattr(self._tools, name, None)
        is_tool = kind == "tool"
        if target is None:
            self._problem(
                "missing_tool" if is_tool else "missing_handler",
                f"{where} references {kind} '{name}', which is not in the tools",
                node=node,
                function=name if is_tool else None,
            )
            return None
        if not callable(target):
            self._problem(
                "invalid_tool" if is_tool else "missing_handler",
                f"{where} {kind} '{name}' is not callable",
                node=node,
                function=name if is_tool else None,
            )
            return None
        return target

    def _render(self, text: str, where: str) -> str:
        """Substitute variables into ``text``, recording any without a value."""

        def substitute(match: re.Match) -> str:
            name = match.group(1)
            if name not in self._variables:
                self._problem(
                    "missing_variable", f"{where} uses variable '{name}', which has no value"
                )
                return match.group(0)
            return str(self._variables[name])

        return _VARIABLE.sub(substitute, text)

    def _problem(
        self, code: str, message: str, *, node: str | None = None, function: str | None = None
    ) -> None:
        self._problems.append(FlowProblem(code, message, node=node, function=function))

    # Calling

    def _make_handler(
        self, ref: "FlowConfig.Function", wrapper: FlowsDirectFunctionWrapper, where: str
    ) -> Callable[[FlowArgs, "FlowManager"], Any]:
        async def handler(
            args: FlowArgs, flow_manager: "FlowManager"
        ) -> ConsolidatedFunctionResult:
            response = await wrapper.invoke(args, flow_manager)
            result, next_node = _check_contract(response, ref.name, where)
            if next_node is NO_RESPONSE:
                return result, NO_RESPONSE
            return result, self._destination(ref, result, where)

        return handler

    def _destination(
        self, ref: "FlowConfig.Function", result: Any, where: str
    ) -> NodeConfig | None:
        target = ref.transition_to
        if target is None:
            return None
        if isinstance(target, str):
            return self._nodes[target]

        if not isinstance(result, Mapping) or target.field not in result:
            raise FlowError(
                f"{where} tool '{ref.name}' branches on result field '{target.field}', "
                f"but its result has no such field"
            )
        value = result[target.field]
        key = value if isinstance(value, str) else str(value)
        node_name = target.cases.get(key, target.default)
        if node_name is None:
            logger.debug(f"{where} tool '{ref.name}': no branch case for {value!r}, staying")
            return None
        return self._nodes[node_name]


def _check_contract(response: Any, name: str, where: str) -> tuple[Any, Any]:
    """Check a tool's return value against the configured-flow contract."""
    if not isinstance(response, tuple) or len(response) != 2:
        raise FlowError(
            f"{where} tool '{name}' must return a (result, None) tuple; "
            f"got {type(response).__name__}"
        )
    result, next_node = response
    if next_node is None or next_node is NO_RESPONSE:
        return result, next_node
    if isinstance(next_node, str):
        raise FlowError(
            f"{where} tool '{name}' returned node name '{next_node}'; "
            "in a configured flow the config owns transitions, so return (result, None) "
            "and set transition_to in the config"
        )
    raise FlowError(
        f"{where} tool '{name}' returned a next node; "
        "in a configured flow the config owns transitions, so return (result, None)"
    )


def _warn_if_annotated_with_node(function: Callable, name: str, where: str) -> None:
    """Log when a tool's return annotation says it always picks its own node.

    Only a second tuple element of exactly ``NodeConfig`` triggers the warning.
    ``NodeConfig | None`` and the ``ConsolidatedFunctionResult`` alias admit
    ``None``, so a tool annotated with either may still honor the contract.
    """
    try:
        annotation = inspect.signature(function).return_annotation
    except (TypeError, ValueError):
        return
    if isinstance(annotation, str):
        picks_node = re.search(r",\s*NodeConfig\s*\]$", annotation) is not None
    else:
        args = typing.get_args(annotation)
        picks_node = len(args) == 2 and args[1] is NodeConfig
    if picks_node:
        logger.warning(
            f"{where} tool '{name}' is annotated as returning a NodeConfig; "
            "in a configured flow it must return (result, None)"
        )
