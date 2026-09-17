#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Move between Flows nodes on a TypeSafe judgment instead of an LLM tool call.

Requires the ``typesafe`` extra: ``uv add "pipecat-ai[typesafe]"``.
"""

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from loguru import logger

from pipecat.flows.types import NodeConfig
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.typesafe_choice_router import ROUTE_QUESTION_ID, TypeSafeChoiceRouter
from pipecat.services.typesafe.judge import JudgeResult, TypeSafeJudge

if TYPE_CHECKING:
    from pipecat.flows.manager import FlowManager

RouteTarget = (
    NodeConfig
    | Callable[[], NodeConfig]
    | Callable[["FlowManager", JudgeResult], Awaitable[NodeConfig | None]]
)
"""Where a route leads: a node, a factory for one, or an async handler that may return one."""

STATE_KEY = "typesafe"
"""Key under which the router stores its latest judgment in ``FlowManager.state``."""


@dataclass
class TypeSafeRoute:
    """One option the router can pick, and where it leads.

    Parameters:
        description: What user replies this option covers. Sent to TypeSafe as
            the option's criteria, so write it the way you would explain the
            option to a person.
        target: The node to move to. A ``NodeConfig`` or a no-argument factory
            moves there directly. An async handler taking the flow manager and
            the judgment may do work first and return the node, or return None
            to leave the turn to the LLM.
    """

    description: str
    target: RouteTarget


class FlowsTypeSafeRouter(TypeSafeChoiceRouter):
    """Routes between Flows nodes on a TypeSafe ``Choice``, skipping the LLM.

    Place it between ``context_aggregator.user()`` and the LLM. While the flow
    is in one of ``active_nodes``, each user turn is judged against the
    ``routes``; a confident match moves the flow to the route's target without
    running the LLM. Anything else, including the fallback option and
    judgments below the confidence threshold, continues to the LLM so the
    node's prompt can handle it.

    The nodes you route to should usually set ``respond_immediately=False``
    and speak through a ``tts_say`` or ``end_conversation`` pre-action, so the
    whole exchange stays LLM-free. A target node that responds immediately also
    works: its ``LLMRunFrame`` produces a fresh context frame which passes the
    router because the flow has left the active node.

    The latest judgment is stored in ``flow_manager.state["typesafe"]`` as a
    dict, so a later prompt can read ``{{ typesafe.choices.route.choice }}``
    and handlers can inspect the probabilities.

    Example::

        router = FlowsTypeSafeRouter(
            judge=TypeSafeJudge(),
            active_nodes={"verify"},
            instructions="How did the user answer the question in `bot_message`? Judge `user_reply`.",
            routes={
                "yes": TypeSafeRoute("The user confirms they are over 18", create_confirmed_node),
                "no": TypeSafeRoute("The user says they are not over 18", create_declined_node),
            },
        )
        pipeline = Pipeline([..., context_aggregator.user(), router, llm, ...])
        flow_manager = FlowManager(...)
        router.flow_manager = flow_manager
    """

    def __init__(
        self,
        *,
        judge: TypeSafeJudge,
        instructions: str,
        routes: dict[str, TypeSafeRoute],
        active_nodes: set[str],
        flow_manager: "FlowManager | None" = None,
        **kwargs,
    ):
        """Initialize the router.

        Args:
            judge: The TypeSafe client.
            instructions: The routing question. Refer to the state fields
                ``bot_message`` and ``user_reply`` in backticks.
            routes: Options keyed by option id, each with a description and a
                target.
            active_nodes: Names of the nodes in which turns are judged. Set
                ``name`` on those ``NodeConfig``s.
            flow_manager: The flow manager, when it already exists. Usually it
                is built after the pipeline, so set the :attr:`flow_manager`
                property afterwards instead.
            **kwargs: Additional arguments passed to
                :class:`~pipecat.processors.typesafe_choice_router.TypeSafeChoiceRouter`,
                such as ``confidence_threshold`` or ``extra_questions``.
        """
        self._routes = routes
        self._active_nodes = active_nodes
        self._flow_manager = flow_manager
        super().__init__(
            judge=judge,
            instructions=instructions,
            criteria={option: route.description for option, route in routes.items()},
            on_choice=self._route,
            is_active=self._in_active_node,
            **kwargs,
        )

    @property
    def flow_manager(self) -> "FlowManager | None":
        """The flow manager whose nodes this router moves between."""
        return self._flow_manager

    @flow_manager.setter
    def flow_manager(self, flow_manager: "FlowManager") -> None:
        self._flow_manager = flow_manager

    def _in_active_node(self) -> bool:
        return (
            self._flow_manager is not None and self._flow_manager.current_node in self._active_nodes
        )

    async def _route(self, result: JudgeResult, context: LLMContext) -> bool:
        flow_manager = self._flow_manager
        if flow_manager is None:
            return False
        option = result.choices[ROUTE_QUESTION_ID].choice
        route = self._routes.get(option)
        if route is None:
            logger.warning(f"{self}: TypeSafe chose {option!r}, which has no route")
            return False

        flow_manager.state[STATE_KEY] = result.model_dump()

        node = await self._resolve_target(route.target, flow_manager, result)
        if node is None:
            logger.debug(f"{self}: route {option!r} left the turn to the LLM")
            return False

        logger.info(f"{self}: routed {option!r} to node {node.get('name')!r} without the LLM")
        await flow_manager.set_node_from_config(node)
        return True

    @staticmethod
    async def _resolve_target(
        target: RouteTarget, flow_manager: "FlowManager", result: JudgeResult
    ) -> NodeConfig | None:
        if isinstance(target, dict):
            return cast(NodeConfig, target)
        if inspect.iscoroutinefunction(target):
            return await target(flow_manager, result)
        return cast(Callable[[], NodeConfig], target)()
