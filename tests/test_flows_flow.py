#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from loguru import logger

from pipecat.flows import (
    NO_RESPONSE,
    ContextStrategy,
    ContextStrategyConfig,
    Flow,
    FlowConfig,
    FlowError,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig,
    flows_tool_options,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from tests.flows_test_helpers import get_advertised_tool_handlers, make_mock_worker

# --- A tools module, as an application would write it ---


async def choose_pizza(flow_manager):
    """User wants to order pizza."""
    return None, None


@flows_tool_options(cancel_on_interruption=True, timeout_secs=12)
async def select_pizza_order(flow_manager, size: str, pizza_type: str):
    """Record the pizza order details.

    Args:
        size (str): One of "small", "medium", or "large".
        pizza_type (str): The kind of pizza.
    """
    flow_manager.state["order"] = {"size": size, "pizza_type": pizza_type}
    return {"size": size, "type": pizza_type, "status": "ok"}, None


async def report(flow_manager, status: str):
    """Report a status the config can branch on.

    Args:
        status (str): The status to report.
    """
    return {"status": status}, None


async def report_flag(flow_manager, flag: bool):
    """Report a boolean the config can branch on.

    Args:
        flag (bool): The flag to report.
    """
    return {"flag": flag}, None


async def get_delivery_estimate(flow_manager):
    """Get a delivery estimate."""
    return {"time": "30 minutes"}, None


async def stay_quiet(flow_manager):
    """Do work without prompting a bot response."""
    return {"ok": True}, NO_RESPONSE


async def returns_bare(flow_manager):
    """Break the contract with a bare result."""
    return {"ok": True}


async def returns_node(flow_manager):
    """Break the contract with a node."""
    return None, {"task_messages": []}


async def returns_name(flow_manager):
    """Break the contract with a node name."""
    return None, "confirm"


async def returns_non_mapping(flow_manager):
    """Return a result a branch cannot read."""
    return "sold out", None


async def annotated_for_hand_built(flow_manager) -> tuple[None, NodeConfig]:
    """A tool annotated as choosing its own node."""
    return None, None


def not_async(flow_manager):
    """Not a coroutine."""
    return None, None


async def wrong_first_param(fm):
    """Wrong first parameter name."""
    return None, None


kitchen_checks: list[dict] = []


async def check_kitchen_status(action: dict, flow_manager) -> None:
    """Pre-action handler."""
    kitchen_checks.append(action)


NOT_A_TOOL = "just a constant"

TOOLS = SimpleNamespace(**{k: v for k, v in globals().items() if not k.startswith("_")})


def config(**overrides) -> FlowConfig:
    data = {
        "initial_node": "initial",
        "nodes": {
            "initial": {
                "role_message": "You work for {{ restaurant }}.",
                "task_messages": [{"role": "developer", "content": "Greet the {{ caller }}."}],
                "pre_actions": [{"type": "function", "handler": "check_kitchen_status"}],
                "functions": [{"name": "choose_pizza", "transition_to": "pizza"}],
            },
            "pizza": {
                "task_messages": [{"role": "developer", "content": "Take the order."}],
                "context_strategy": "reset",
                "respond_immediately": False,
                "functions": [{"name": "select_pizza_order", "transition_to": "confirm"}],
                "post_actions": [{"type": "tts_say", "text": "Thanks, {{ caller }}!"}],
            },
            "confirm": {
                "task_messages": [{"role": "developer", "content": "Confirm."}],
            },
        },
        "global_functions": [{"name": "get_delivery_estimate"}],
    }
    data.update(overrides)
    return FlowConfig.model_validate(data)


def single_node(function: dict, **node_extras) -> FlowConfig:
    """A config with one node offering one function, plus a 'next' node."""
    return FlowConfig.model_validate(
        {
            "initial_node": "a",
            "nodes": {
                "a": {
                    "task_messages": [{"role": "developer", "content": "a"}],
                    "functions": [function],
                    **node_extras,
                },
                "next": {"task_messages": [{"role": "developer", "content": "next"}]},
                "other": {"task_messages": [{"role": "developer", "content": "other"}]},
            },
        }
    )


VARIABLES = {"restaurant": "Luigi's", "caller": "friend"}


def bind(cfg: FlowConfig, tools=TOOLS, variables=VARIABLES) -> Flow:
    return cfg.bind(tools=tools, variables=variables)


class TestBinding(unittest.TestCase):
    def test_bind_returns_flow(self):
        cfg = config()
        flow = bind(cfg)
        self.assertIsInstance(flow, Flow)
        self.assertIs(flow.config, cfg)

    def test_tools_from_mapping(self):
        mapping = {
            "choose_pizza": choose_pizza,
            "select_pizza_order": select_pizza_order,
            "get_delivery_estimate": get_delivery_estimate,
            "check_kitchen_status": check_kitchen_status,
            "unrelated": 42,
        }
        flow = bind(config(), tools=mapping)
        self.assertEqual(flow.initial_node["functions"][0].name, "choose_pizza")

    def test_missing_tool(self):
        cfg = single_node({"name": "no_such_tool"})
        with self.assertRaises(FlowError) as cm:
            bind(cfg)
        self.assertIn("node 'a' references tool 'no_such_tool'", str(cm.exception))

    def test_missing_global_tool(self):
        cfg = config(global_functions=[{"name": "no_such_tool"}])
        with self.assertRaises(FlowError) as cm:
            bind(cfg)
        self.assertIn("global_functions references tool 'no_such_tool'", str(cm.exception))

    def test_missing_action_handler(self):
        cfg = single_node(
            {"name": "choose_pizza"},
            pre_actions=[{"type": "function", "handler": "no_such_handler"}],
        )
        with self.assertRaises(FlowError) as cm:
            bind(cfg)
        self.assertIn("pre_actions references action handler 'no_such_handler'", str(cm.exception))

    def test_non_callable_reference(self):
        with self.assertRaises(FlowError) as cm:
            bind(single_node({"name": "NOT_A_TOOL"}))
        self.assertIn("is not callable", str(cm.exception))

    def test_invalid_direct_function_not_async(self):
        with self.assertRaises(FlowError) as cm:
            bind(single_node({"name": "not_async"}))
        self.assertIn("not a valid direct function", str(cm.exception))
        self.assertIn("must be async", str(cm.exception))

    def test_invalid_direct_function_first_param(self):
        with self.assertRaises(FlowError) as cm:
            bind(single_node({"name": "wrong_first_param"}))
        self.assertIn("flow_manager", str(cm.exception))

    def test_missing_variable(self):
        with self.assertRaises(FlowError) as cm:
            bind(config(), variables={"restaurant": "Luigi's"})
        self.assertIn("node 'initial' task_messages uses variable 'caller'", str(cm.exception))

    def test_variables_render_everywhere(self):
        flow = bind(config(), variables={"restaurant": "Luigi's", "caller": 7})
        initial, pizza = flow.node("initial"), flow.node("pizza")
        self.assertEqual(initial["role_message"], "You work for Luigi's.")
        self.assertEqual(initial["task_messages"][0]["content"], "Greet the 7.")
        self.assertEqual(pizza["post_actions"][0]["text"], "Thanks, 7!")

    def test_unused_variables_are_fine(self):
        bind(config(), variables={**VARIABLES, "extra": "unused"})

    def test_annotation_warning(self):
        records = []
        sink = logger.add(lambda m: records.append(m.record["message"]), level="WARNING")
        try:
            bind(single_node({"name": "annotated_for_hand_built"}))
        finally:
            logger.remove(sink)
        self.assertTrue(any("annotated as returning a NodeConfig" in r for r in records))


class TestNodeConfigs(unittest.TestCase):
    def setUp(self):
        self.flow = bind(config())

    def test_initial_node(self):
        self.assertIs(self.flow.initial_node, self.flow.node("initial"))

    def test_unknown_node(self):
        with self.assertRaises(FlowError):
            self.flow.node("nope")

    def test_node_shape(self):
        pizza = self.flow.node("pizza")
        self.assertEqual(pizza["name"], "pizza")
        self.assertEqual(
            pizza["task_messages"], [{"role": "developer", "content": "Take the order."}]
        )
        self.assertFalse(pizza["respond_immediately"])
        self.assertEqual(
            pizza["context_strategy"], ContextStrategyConfig(strategy=ContextStrategy.RESET)
        )
        self.assertNotIn("role_message", pizza)
        self.assertNotIn("pre_actions", pizza)

        initial = self.flow.node("initial")
        self.assertTrue(initial["respond_immediately"])
        self.assertNotIn("context_strategy", initial)

        confirm = self.flow.node("confirm")
        self.assertNotIn("functions", confirm)

    def test_function_schema_from_direct_function(self):
        schema = self.flow.node("pizza")["functions"][0]
        self.assertIsInstance(schema, FlowsFunctionSchema)
        self.assertEqual(schema.name, "select_pizza_order")
        self.assertEqual(schema.description, "Record the pizza order details.")
        self.assertEqual(set(schema.properties), {"size", "pizza_type"})
        self.assertEqual(
            schema.properties["size"]["description"], 'One of "small", "medium", or "large".'
        )
        self.assertEqual(schema.required, ["size", "pizza_type"])
        self.assertTrue(schema.cancel_on_interruption)
        self.assertEqual(schema.timeout_secs, 12)

    def test_undecorated_tool_defaults(self):
        schema = self.flow.node("initial")["functions"][0]
        self.assertFalse(schema.cancel_on_interruption)
        self.assertIsNone(schema.timeout_secs)

    def test_actions(self):
        pre = self.flow.node("initial")["pre_actions"]
        self.assertEqual(pre, [{"type": "function", "handler": check_kitchen_status}])
        post = self.flow.node("pizza")["post_actions"]
        self.assertEqual(post, [{"type": "tts_say", "text": "Thanks, friend!"}])

    def test_global_functions(self):
        globals_ = self.flow.global_functions
        self.assertEqual([g.name for g in globals_], ["get_delivery_estimate"])
        self.assertIsNot(globals_, self.flow.global_functions)


class TestCallTimeContract(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.flow_manager = SimpleNamespace(state={})

    async def call(self, flow: Flow, node: str = "a", args: dict | None = None):
        schema = flow.node(node)["functions"][0]
        return await schema.handler(args or {}, self.flow_manager)

    async def test_none_with_transition_uses_config_edge(self):
        flow = bind(single_node({"name": "choose_pizza", "transition_to": "next"}))
        result, next_node = await self.call(flow)
        self.assertIsNone(result)
        self.assertIs(next_node, flow.node("next"))

    async def test_none_without_transition_stays(self):
        flow = bind(single_node({"name": "get_delivery_estimate"}))
        result, next_node = await self.call(flow)
        self.assertEqual(result, {"time": "30 minutes"})
        self.assertIsNone(next_node)

    async def test_arguments_reach_the_tool(self):
        flow = bind(single_node({"name": "select_pizza_order", "transition_to": "next"}))
        result, next_node = await self.call(flow, args={"size": "large", "pizza_type": "cheese"})
        self.assertEqual(result["size"], "large")
        self.assertEqual(self.flow_manager.state["order"]["pizza_type"], "cheese")
        self.assertIs(next_node, flow.node("next"))

    async def test_no_response_passes_through(self):
        flow = bind(single_node({"name": "stay_quiet", "transition_to": "next"}))
        result, next_node = await self.call(flow)
        self.assertEqual(result, {"ok": True})
        self.assertIs(next_node, NO_RESPONSE)

    async def test_bare_result_is_rejected(self):
        flow = bind(single_node({"name": "returns_bare"}))
        with self.assertRaises(FlowError) as cm:
            await self.call(flow)
        self.assertIn("must return a (result, None) tuple", str(cm.exception))

    async def test_node_return_is_rejected(self):
        for function in (
            {"name": "returns_node"},
            {"name": "returns_node", "transition_to": "next"},
        ):
            flow = bind(single_node(function))
            with self.assertRaises(FlowError) as cm:
                await self.call(flow)
            self.assertIn("config owns transitions", str(cm.exception))

    async def test_name_return_is_rejected(self):
        flow = bind(single_node({"name": "returns_name"}))
        with self.assertRaises(FlowError) as cm:
            await self.call(flow)
        self.assertIn("returned node name 'confirm'", str(cm.exception))


class TestBranches(unittest.IsolatedAsyncioTestCase):
    BRANCH = {"field": "status", "cases": {"ok": "next", "sold_out": "other"}}

    def setUp(self):
        self.flow_manager = SimpleNamespace(state={})

    async def branch(self, tool: str, transition_to: dict, args: dict):
        flow = bind(single_node({"name": tool, "transition_to": transition_to}))
        schema = flow.node("a")["functions"][0]
        _, next_node = await schema.handler(args, self.flow_manager)
        return flow, next_node

    async def test_case_match(self):
        flow, next_node = await self.branch("report", self.BRANCH, {"status": "sold_out"})
        self.assertIs(next_node, flow.node("other"))

    async def test_default(self):
        flow, next_node = await self.branch(
            "report", {**self.BRANCH, "default": "next"}, {"status": "weird"}
        )
        self.assertIs(next_node, flow.node("next"))

    async def test_no_match_no_default_stays(self):
        _, next_node = await self.branch("report", self.BRANCH, {"status": "weird"})
        self.assertIsNone(next_node)

    async def test_non_string_values_match_by_str(self):
        flow, next_node = await self.branch(
            "report_flag", {"field": "flag", "cases": {"True": "next"}}, {"flag": True}
        )
        self.assertIs(next_node, flow.node("next"))

    async def test_missing_field_is_an_error(self):
        with self.assertRaises(FlowError) as cm:
            await self.branch(
                "report", {"field": "missing", "cases": {"x": "next"}}, {"status": "ok"}
            )
        self.assertIn("branches on result field 'missing'", str(cm.exception))

    async def test_non_mapping_result_is_an_error(self):
        with self.assertRaises(FlowError) as cm:
            await self.branch("returns_non_mapping", self.BRANCH, {})
        self.assertIn("has no such field", str(cm.exception))


class TestWithFlowManager(unittest.IsolatedAsyncioTestCase):
    """A bound flow driven through the real FlowManager."""

    async def asyncSetUp(self):
        self.mock_worker = make_mock_worker()
        assistant = MagicMock()
        type(assistant).has_function_calls_in_progress = PropertyMock(return_value=False)
        self.context_aggregator = MagicMock()
        self.context_aggregator.user = MagicMock(return_value=MagicMock())
        self.context_aggregator.assistant = MagicMock(return_value=assistant)
        self.llm = OpenAILLMService(api_key="test-key")
        kitchen_checks.clear()

    def make_manager(self, flow: Flow) -> FlowManager:
        return FlowManager(
            worker=self.mock_worker,
            llm=self.llm,
            context_aggregator=self.context_aggregator,
            global_functions=flow.global_functions,
        )

    async def invoke(self, name: str, arguments: dict | None = None):
        """Call an advertised tool the way the LLM service would, then let the
        context-updated callback fire so a pending transition executes."""
        handler = get_advertised_tool_handlers(self.mock_worker)[name]
        results = []

        async def result_callback(result, *, properties=None):
            results.append(result)
            if properties is not None and properties.on_context_updated is not None:
                await properties.on_context_updated()

        await handler(
            FunctionCallParams(
                function_name=name,
                tool_call_id="t1",
                arguments=arguments or {},
                llm=self.llm,
                pipeline_worker=self.mock_worker,
                context=None,
                result_callback=result_callback,
            )
        )
        return results[0]

    async def test_configured_flow_end_to_end(self):
        flow = bind(config())
        flow_manager = self.make_manager(flow)
        await flow_manager.initialize(flow.initial_node)

        self.assertEqual(flow_manager.current_node, "initial")
        self.assertEqual(len(kitchen_checks), 1)
        self.assertEqual(
            set(get_advertised_tool_handlers(self.mock_worker)),
            {"choose_pizza", "get_delivery_estimate"},
        )

        result = await self.invoke("choose_pizza")
        self.assertEqual(result, {"status": "acknowledged"})
        self.assertEqual(flow_manager.current_node, "pizza")
        self.assertEqual(
            set(get_advertised_tool_handlers(self.mock_worker)),
            {"select_pizza_order", "get_delivery_estimate"},
        )

        result = await self.invoke("select_pizza_order", {"size": "small", "pizza_type": "cheese"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(flow_manager.state["order"]["size"], "small")
        self.assertEqual(flow_manager.current_node, "confirm")

        result = await self.invoke("get_delivery_estimate")
        self.assertEqual(result, {"time": "30 minutes"})
        self.assertEqual(flow_manager.current_node, "confirm")

    async def test_contract_violation_reaches_llm_as_error(self):
        flow = bind(single_node({"name": "returns_bare"}))
        flow_manager = self.make_manager(flow)
        await flow_manager.initialize(flow.initial_node)

        result = await self.invoke("returns_bare")
        self.assertEqual(result["status"], "error")
        self.assertIn("must return a (result, None) tuple", result["error"])
        self.assertEqual(flow_manager.current_node, "a")

    async def test_same_tool_works_in_a_hand_built_flow(self):
        flow_manager = FlowManager(
            worker=self.mock_worker, llm=self.llm, context_aggregator=self.context_aggregator
        )
        await flow_manager.initialize()
        node: NodeConfig = {
            "name": "hand_built",
            "task_messages": [{"role": "developer", "content": "Order."}],
            "functions": [select_pizza_order],
        }
        await flow_manager.set_node_from_config(node)

        result = await self.invoke("select_pizza_order", {"size": "large", "pizza_type": "veg"})
        self.assertEqual(result["size"], "large")
        self.assertEqual(flow_manager.current_node, "hand_built")


if __name__ == "__main__":
    unittest.main()
