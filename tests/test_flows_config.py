#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

import pipecat.flows
from pipecat.flows import ContextStrategy, FlowConfig

FOOD_ORDERING = """
initial_node: initial

nodes:
  initial:
    role_message: You are an order-taking assistant for {{ restaurant_name }}.
    task_messages:
      - role: developer
        content: Greet the caller and ask whether they want pizza or sushi.
    pre_actions:
      - type: function
        handler: check_kitchen_status
    functions:
      - name: choose_pizza
        transition_to: choose_pizza
      - name: choose_sushi
        transition_to: choose_sushi

  choose_pizza:
    task_messages:
      - role: developer
        content: Take a pizza order.
    functions:
      - name: select_pizza_order
        transition_to:
          field: status
          cases:
            ok: confirm
            unavailable: choose_pizza
          default: confirm

  choose_sushi:
    task_messages:
      - role: developer
        content: Take a sushi order.
    context_strategy: reset
    respond_immediately: false
    functions:
      - name: select_sushi_order
        transition_to: confirm

  confirm:
    task_messages:
      - role: developer
        content: Read the order back.
    functions:
      - name: complete_order
        transition_to: end
      - name: revise_order
        transition_to: initial

  end:
    task_messages:
      - role: developer
        content: Thank the caller.
    post_actions:
      - type: tts_say
        text: Goodbye!
      - type: end_conversation

global_functions:
  - name: get_delivery_estimate
"""


def _minimal(**overrides) -> dict:
    data = {
        "initial_node": "a",
        "nodes": {"a": {"task_messages": [{"role": "developer", "content": "hi"}]}},
    }
    data.update(overrides)
    return data


class TestFlowConfigLoading(unittest.TestCase):
    def test_loads_food_ordering_yaml(self):
        config = FlowConfig.from_yaml(FOOD_ORDERING)

        self.assertEqual(config.initial_node, "initial")
        self.assertEqual(
            set(config.nodes), {"initial", "choose_pizza", "choose_sushi", "confirm", "end"}
        )

        initial = config.nodes["initial"]
        self.assertIn("{{ restaurant_name }}", initial.role_message)
        self.assertEqual([f.name for f in initial.functions], ["choose_pizza", "choose_sushi"])
        self.assertEqual(initial.functions[0].transition_to, "choose_pizza")
        self.assertEqual(initial.pre_actions[0].type, "function")
        self.assertEqual(initial.pre_actions[0].handler, "check_kitchen_status")

        pizza = config.nodes["choose_pizza"].functions[0]
        self.assertIsInstance(pizza.transition_to, FlowConfig.Branch)
        self.assertEqual(pizza.transition_to.field, "status")
        self.assertEqual(
            pizza.transition_to.cases, {"ok": "confirm", "unavailable": "choose_pizza"}
        )
        self.assertEqual(pizza.transition_to.default, "confirm")

        sushi = config.nodes["choose_sushi"]
        self.assertEqual(sushi.context_strategy, "reset")
        self.assertEqual(sushi.context_strategy_enum(), ContextStrategy.RESET)
        self.assertFalse(sushi.respond_immediately)

        end = config.nodes["end"]
        self.assertEqual(end.post_actions[0].type, "tts_say")
        self.assertEqual(end.post_actions[0].extras(), {"text": "Goodbye!"})
        self.assertEqual(end.post_actions[1].type, "end_conversation")

        self.assertEqual([f.name for f in config.global_functions], ["get_delivery_estimate"])
        self.assertIsNone(config.global_functions[0].transition_to)

    def test_defaults(self):
        config = FlowConfig.model_validate(_minimal())
        node = config.nodes["a"]
        self.assertIsNone(node.role_message)
        self.assertEqual(config.global_functions, [])
        self.assertEqual(node.functions, [])
        self.assertEqual(node.pre_actions, [])
        self.assertEqual(node.post_actions, [])
        self.assertIsNone(node.context_strategy)
        self.assertIsNone(node.context_strategy_enum())
        self.assertTrue(node.respond_immediately)

    def test_json_round_trip_matches_yaml(self):
        from_yaml = FlowConfig.from_yaml(FOOD_ORDERING)
        from_json = FlowConfig.from_json(json.dumps(from_yaml.model_dump()))
        self.assertEqual(from_json, from_yaml)

    def test_from_file_yaml_with_include(self):
        d = Path(tempfile.mkdtemp())
        (d / "greeting.yaml").write_text(
            "- role: developer\n  content: Greet the caller.\n", encoding="utf-8"
        )
        (d / "flow.yaml").write_text(
            "initial_node: a\nnodes:\n  a:\n    task_messages: !include greeting.yaml\n",
            encoding="utf-8",
        )

        config = FlowConfig.from_file(d / "flow.yaml")
        self.assertEqual(config.nodes["a"].task_messages[0].content, "Greet the caller.")

    def test_from_file_json(self):
        d = Path(tempfile.mkdtemp())
        (d / "flow.json").write_text(json.dumps(_minimal()), encoding="utf-8")
        config = FlowConfig.from_file(d / "flow.json")
        self.assertEqual(config.initial_node, "a")

    def test_from_json_requires_a_mapping(self):
        with self.assertRaises(ValueError):
            FlowConfig.from_json('["just", "a", "list"]')

    def test_from_yaml_without_base_dir_rejects_include(self):
        with self.assertRaises(yaml.constructor.ConstructorError):
            FlowConfig.from_yaml("initial_node: a\nnodes: !include nodes.yaml\n")

    def test_top_level_must_be_mapping(self):
        with self.assertRaises(ValueError) as cm:
            FlowConfig.from_yaml("- just\n- a list\n")
        self.assertIn("top level must be a mapping", str(cm.exception))

    def test_json_schema_is_exportable(self):
        schema = FlowConfig.model_json_schema()
        self.assertEqual(schema["required"], ["initial_node", "nodes"])
        self.assertIn("Branch", schema["$defs"])

    def test_json_schema_has_optional_metadata(self):
        schema = FlowConfig.model_json_schema()
        self.assertEqual(schema["required"], ["initial_node", "nodes"])
        for model in ("Node", "Function"):
            self.assertIn("metadata", schema["$defs"][model]["properties"])
            self.assertNotIn("metadata", schema["$defs"][model].get("required", []))

    def test_json_schema_has_optional_subflows(self):
        schema = FlowConfig.model_json_schema()
        # The root stays an inline object schema, not a reference.
        self.assertNotIn("$ref", schema)
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["initial_node", "nodes"])
        self.assertIn("flows", schema["properties"])
        self.assertEqual(schema["$defs"]["Subflow"]["required"], ["initial_node", "nodes"])
        node = schema["$defs"]["Node"]
        self.assertIn("flow", node["properties"])
        self.assertIn("return_to", node["properties"])
        # A node needs task_messages unless it runs a flow.
        self.assertNotIn("required", node)
        self.assertEqual(node["else"]["required"], ["task_messages"])

    def test_shipped_schema_file_is_current(self):
        # The repository keeps the schema for editors and other tools to vendor.
        # Regenerate it with `uv run python scripts/flows/write_flow_config_schema.py`.
        path = Path(pipecat.flows.__file__).parent / "flow_config.schema.json"
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            **FlowConfig.model_json_schema(),
        }
        expected = json.dumps(schema, indent=2) + "\n"
        self.assertEqual(
            path.read_text(encoding="utf-8"),
            expected,
            "flow_config.schema.json is out of date; run "
            "`uv run python scripts/flows/write_flow_config_schema.py`",
        )


class TestFlowConfigValidation(unittest.TestCase):
    def assert_invalid(self, data: dict, message: str) -> None:
        with self.assertRaises(ValidationError) as cm:
            FlowConfig.model_validate(data)
        self.assertIn(message, str(cm.exception))

    def test_initial_node_must_exist(self):
        self.assert_invalid(_minimal(initial_node="missing"), "initial_node 'missing'")

    def test_unknown_top_level_key(self):
        self.assert_invalid(_minimal(extra_key=1), "extra_key")

    def test_nodes_required_and_non_empty(self):
        self.assert_invalid({"initial_node": "a", "nodes": {}}, "nodes")

    def test_task_messages_required(self):
        self.assert_invalid({"initial_node": "a", "nodes": {"a": {}}}, "task_messages")

    def test_transition_to_unknown_node(self):
        data = _minimal()
        data["nodes"]["a"]["functions"] = [{"name": "go", "transition_to": "nowhere"}]
        self.assert_invalid(data, "node 'a' function 'go' transitions to unknown node 'nowhere'")

    def test_branch_case_to_unknown_node(self):
        data = _minimal()
        data["nodes"]["a"]["functions"] = [
            {"name": "go", "transition_to": {"field": "s", "cases": {"x": "nowhere"}}}
        ]
        self.assert_invalid(data, "unknown node 'nowhere'")

    def test_branch_default_to_unknown_node(self):
        data = _minimal()
        data["nodes"]["a"]["functions"] = [
            {"name": "go", "transition_to": {"field": "s", "cases": {"x": "a"}, "default": "zz"}}
        ]
        self.assert_invalid(data, "unknown node 'zz'")

    def test_branch_keys_are_canonical_strings(self):
        data = _minimal()
        data["nodes"]["a"]["functions"] = [
            {"name": "go", "transition_to": {"field": "s", "cases": {True: "a", 3: "a", "x": "a"}}}
        ]
        cases = FlowConfig.model_validate(data).nodes["a"].functions[0].transition_to.cases
        self.assertEqual(cases, {"true": "a", "3": "a", "x": "a"})

    def test_yaml_boolean_and_number_keys(self):
        config = FlowConfig.from_yaml(
            "initial_node: a\nnodes:\n  a:\n    task_messages: [{role: developer, content: a}]\n"
            "    functions:\n      - name: go\n        transition_to:\n          field: ok\n"
            "          cases:\n            true: a\n            0: a\n"
        )
        self.assertEqual(
            config.nodes["a"].functions[0].transition_to.cases, {"true": "a", "0": "a"}
        )

    def test_branch_requires_cases(self):
        data = _minimal()
        data["nodes"]["a"]["functions"] = [
            {"name": "go", "transition_to": {"field": "s", "cases": {}}}
        ]
        self.assert_invalid(data, "cases")

    def test_global_function_transition_to_unknown_node(self):
        data = _minimal(global_functions=[{"name": "go", "transition_to": "nowhere"}])
        self.assert_invalid(data, "global_functions function 'go' transitions to unknown node")

    def test_duplicate_function_in_node(self):
        data = _minimal()
        data["nodes"]["a"]["functions"] = [{"name": "f"}, {"name": "f"}]
        self.assert_invalid(data, "duplicate function 'f' in node")

    def test_duplicate_global_function(self):
        data = _minimal(global_functions=[{"name": "f"}, {"name": "f"}])
        self.assert_invalid(data, "duplicate function 'f' in global_functions")

    def test_node_function_collides_with_global(self):
        data = _minimal(global_functions=[{"name": "f"}])
        data["nodes"]["a"]["functions"] = [{"name": "f"}]
        self.assert_invalid(data, "node 'a' function 'f' is also a global function")

    def test_function_rejects_unknown_keys(self):
        data = _minimal()
        data["nodes"]["a"]["functions"] = [{"name": "f", "parameters": {}}]
        self.assert_invalid(data, "parameters")

    def test_description_only_on_transition_only_function(self):
        data = _minimal()
        data["nodes"]["a"]["functions"] = [{"name": "f", "description": "d", "transition_to": "a"}]
        self.assert_invalid(data, "only a transition_only function takes")

    def test_transition_only_function_needs_description(self):
        data = _minimal()
        data["nodes"]["a"]["functions"] = [
            {"name": "f", "transition_only": True, "transition_to": "a"}
        ]
        self.assert_invalid(data, "needs a description")

    def test_transition_only_function_needs_a_node_name(self):
        data = _minimal()
        data["nodes"]["a"]["functions"] = [
            {"name": "f", "transition_only": True, "description": "d"}
        ]
        self.assert_invalid(data, "must name the node it transitions to")
        data["nodes"]["a"]["functions"] = [
            {
                "name": "f",
                "transition_only": True,
                "description": "d",
                "transition_to": {"field": "x", "cases": {"y": "a"}},
            }
        ]
        self.assert_invalid(data, "must name the node it transitions to")

    def test_metadata_on_node_and_function(self):
        data = _minimal()
        node = data["nodes"]["a"]
        node["metadata"] = {"graph_node": "n1", "position": {"x": 1, "y": 2.5}, "tags": ["a"]}
        node["functions"] = [{"name": "f", "metadata": {"exit": "chosen"}}]
        cfg = FlowConfig.model_validate(data)
        self.assertEqual(cfg.nodes["a"].metadata, node["metadata"])
        self.assertEqual(cfg.nodes["a"].functions[0].metadata, {"exit": "chosen"})

        self.assertEqual(FlowConfig.model_validate_json(cfg.model_dump_json()), cfg)

    def test_metadata_defaults_to_none(self):
        data = _minimal()
        data["nodes"]["a"]["functions"] = [{"name": "f"}]
        cfg = FlowConfig.model_validate(data)
        self.assertIsNone(cfg.nodes["a"].metadata)
        self.assertIsNone(cfg.nodes["a"].functions[0].metadata)

    def test_metadata_must_be_json(self):
        data = _minimal()
        data["nodes"]["a"]["metadata"] = {"handle": object()}
        self.assert_invalid(data, "metadata")

        data = _minimal()
        data["nodes"]["a"]["functions"] = [{"name": "f", "metadata": {"handle": object()}}]
        self.assert_invalid(data, "metadata")

        data = _minimal()
        data["nodes"]["a"]["metadata"] = ["not", "a", "mapping"]
        self.assert_invalid(data, "metadata")

    def test_transition_only_function(self):
        data = _minimal()
        data["nodes"]["a"]["functions"] = [
            {"name": "f", "transition_only": True, "description": "d", "transition_to": "a"}
        ]
        cfg = FlowConfig.model_validate(data)
        func = cfg.nodes["a"].functions[0]
        self.assertTrue(func.transition_only)
        self.assertEqual(func.targets(), ["a"])

    def test_function_action_requires_handler(self):
        data = _minimal()
        data["nodes"]["a"]["pre_actions"] = [{"type": "function"}]
        self.assert_invalid(data, "a 'function' action requires a 'handler' name")

    def test_built_in_actions_reject_handler(self):
        for built_in in ("tts_say", "end_conversation"):
            data = _minimal()
            data["nodes"]["a"]["post_actions"] = [{"type": built_in, "handler": "x"}]
            self.assert_invalid(data, f"the built-in '{built_in}' action does not take a 'handler'")

    def test_custom_action_may_name_a_handler(self):
        data = _minimal()
        data["nodes"]["a"]["post_actions"] = [
            {"type": "notify", "handler": "notify_slack", "channel": "#x"},
            {"type": "audit"},
        ]
        actions = FlowConfig.model_validate(data).nodes["a"].post_actions
        self.assertEqual(actions[0].handler, "notify_slack")
        self.assertFalse(actions[0].registered_in_code)
        self.assertTrue(actions[1].registered_in_code)
        self.assertEqual(actions[0].extras(), {"channel": "#x"})

    def test_context_strategy_values(self):
        data = _minimal()
        data["nodes"]["a"]["context_strategy"] = "reset_with_summary"
        self.assert_invalid(data, "context_strategy")


if __name__ == "__main__":
    unittest.main()


def _node(*functions: dict) -> dict:
    node: dict = {"task_messages": [{"role": "developer", "content": "hi"}]}
    if functions:
        node["functions"] = list(functions)
    return node


def _go(name: str, target: str) -> dict:
    return {"name": name, "transition_only": True, "description": "Go.", "transition_to": target}


def _composed() -> dict:
    """A flow whose ``verify`` node runs a verification subflow and comes back to ``book``."""
    return {
        "initial_node": "start",
        "flows": {
            "verification": {
                "initial_node": "ask",
                "nodes": {
                    "ask": _node({"name": "check", "transition_to": "confirm"}),
                    "confirm": _node(_go("verified", "return")),
                },
            },
        },
        "nodes": {
            "start": _node(_go("begin", "verify")),
            "verify": {"flow": "verification", "return_to": "book"},
            "book": _node(),
        },
    }


class TestSubflows(unittest.TestCase):
    def assert_invalid(self, data: dict, *messages: str) -> None:
        with self.assertRaises(ValidationError) as cm:
            FlowConfig.model_validate(data)
        for message in messages:
            self.assertIn(message, str(cm.exception))

    def test_composed_config(self):
        cfg = FlowConfig.model_validate(_composed())
        self.assertEqual(cfg.nodes["verify"].flow, "verification")
        self.assertEqual(cfg.nodes["verify"].return_to, "book")
        self.assertEqual(cfg.nodes["verify"].task_messages, [])
        self.assertEqual(set(cfg.flows["verification"].nodes), {"ask", "confirm"})
        self.assertEqual(FlowConfig.model_validate_json(cfg.model_dump_json()), cfg)
        self.assertEqual(FlowConfig.model_validate(cfg.model_dump()), cfg)

    def test_defaults_to_no_subflows(self):
        cfg = FlowConfig.model_validate(_minimal())
        self.assertEqual(cfg.flows, {})
        self.assertIsNone(cfg.nodes["a"].flow)
        self.assertIsNone(cfg.nodes["a"].return_to)

    def test_subflow_from_an_included_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            subflow = _composed()["flows"]["verification"]
            Path(tmp, "verification.yaml").write_text(yaml.safe_dump(subflow), encoding="utf-8")
            data = _composed()
            del data["flows"]
            text = yaml.safe_dump(data) + "flows:\n  verification: !include verification.yaml\n"
            Path(tmp, "flow.yaml").write_text(text, encoding="utf-8")
            cfg = FlowConfig.from_file(Path(tmp, "flow.yaml"))
        self.assertEqual(cfg, FlowConfig.model_validate(_composed()))

    # A node prompts the LLM or runs a flow, not both.

    def test_node_that_runs_a_flow_takes_nothing_else(self):
        extras = {
            "task_messages": [{"role": "developer", "content": "hi"}],
            "role_message": "You are helpful.",
            "functions": [_go("skip", "book")],
            "pre_actions": [{"type": "tts_say", "text": "One moment."}],
            "post_actions": [{"type": "end_conversation"}],
            "context_strategy": "reset",
            "respond_immediately": False,
        }
        for key, value in extras.items():
            data = _composed()
            data["nodes"]["verify"][key] = value
            self.assert_invalid(
                data, "a node that runs flow 'verification' takes only 'flow', 'return_to'"
            )

    def test_node_that_runs_a_flow_may_have_metadata(self):
        data = _composed()
        data["nodes"]["verify"]["metadata"] = {"graph_node": "n7"}
        cfg = FlowConfig.model_validate(data)
        self.assertEqual(cfg.nodes["verify"].metadata, {"graph_node": "n7"})

    def test_return_to_needs_a_flow(self):
        data = _composed()
        data["nodes"]["book"]["return_to"] = "start"
        self.assert_invalid(data, "'return_to' applies only to a node that runs a 'flow'")

    def test_node_needs_task_messages_or_a_flow(self):
        data = _composed()
        data["nodes"]["book"] = {}
        self.assert_invalid(data, "a node needs 'task_messages', or a 'flow' to run")

    # A node runs a flow the config has.

    def test_node_runs_unknown_flow(self):
        data = _composed()
        data["nodes"]["verify"]["flow"] = "nope"
        self.assert_invalid(data, "node 'verify' runs unknown flow 'nope'")

        plain = _minimal()
        plain["nodes"]["b"] = {"flow": "verification"}
        self.assert_invalid(plain, "node 'b' runs unknown flow 'verification'")

    # return_to names a node, and is needed only when the flow returns.

    def test_return_to_unknown_node(self):
        data = _composed()
        data["nodes"]["verify"]["return_to"] = "nope"
        self.assert_invalid(data, "node 'verify' returns to unknown node 'nope'")

    def test_return_to_a_node_that_runs_a_flow(self):
        data = _composed()
        data["nodes"]["verify_again"] = {"flow": "verification", "return_to": "book"}
        data["nodes"]["verify"]["return_to"] = "verify_again"
        FlowConfig.model_validate(data)

    def test_nodes_may_return_to_each_other(self):
        data = _composed()
        data["nodes"]["verify_again"] = {"flow": "verification", "return_to": "verify"}
        data["nodes"]["verify"]["return_to"] = "verify_again"
        FlowConfig.model_validate(data)

    def test_return_to_is_needed_when_the_flow_returns(self):
        data = _composed()
        del data["nodes"]["verify"]["return_to"]
        self.assert_invalid(
            data, "node 'verify' runs flow 'verification', which returns, and needs a 'return_to'"
        )

    def test_return_to_may_be_omitted_when_the_flow_never_returns(self):
        data = _composed()
        data["flows"]["closing"] = {
            "initial_node": "bye",
            "nodes": {"bye": _node(_go("again", "bye"))},
        }
        data["nodes"]["goodbye"] = {"flow": "closing"}
        cfg = FlowConfig.model_validate(data)
        self.assertIsNone(cfg.nodes["goodbye"].return_to)

        data["nodes"]["goodbye"]["return_to"] = "start"
        FlowConfig.model_validate(data)

    # A node that runs a flow is a node: a target, or where the flow starts.

    def test_initial_node_may_run_a_flow(self):
        data = _composed()
        data["initial_node"] = "verify"
        self.assertEqual(FlowConfig.model_validate(data).initial_node, "verify")

    def test_initial_node_must_exist(self):
        data = _composed()
        data["initial_node"] = "verification"
        self.assert_invalid(data, "initial_node 'verification' is not a defined node")

    def test_global_function_may_enter_a_flow(self):
        data = _composed()
        data["global_functions"] = [_go("reverify", "verify")]
        FlowConfig.model_validate(data)

    def test_branch_may_enter_a_flow(self):
        data = _composed()
        data["nodes"]["book"] = _node(
            {
                "name": "book_slot",
                "transition_to": {
                    "field": "booked",
                    "cases": {"true": "verify"},
                    "default": "book",
                },
            }
        )
        FlowConfig.model_validate(data)

    def test_target_cannot_name_a_flow_or_one_of_its_nodes(self):
        for target in ("verification", "ask", "verify.ask"):
            data = _composed()
            data["nodes"]["book"] = _node(_go("again", target))
            self.assert_invalid(data, f"transitions to unknown node '{target}'")

    def test_return_is_an_ordinary_name_outside_a_subflow(self):
        data = _composed()
        data["nodes"]["book"] = _node(_go("back", "return"))
        self.assert_invalid(
            data, "node 'book' function 'back' transitions to unknown node 'return'"
        )

        data["nodes"]["return"] = _node()
        FlowConfig.model_validate(data)

        plain = _minimal()
        plain["nodes"]["a"]["functions"] = [_go("back", "return")]
        self.assert_invalid(plain, "transitions to unknown node 'return'")
        plain["nodes"]["return"] = _node()
        FlowConfig.model_validate(plain)

    # A target in a subflow names one of its nodes or is `return`.

    def test_subflow_target_must_be_its_own_node_or_return(self):
        for target in ("book", "verify", "nope"):
            data = _composed()
            data["flows"]["verification"]["nodes"]["confirm"] = _node(_go("verified", target))
            self.assert_invalid(
                data,
                "flows.verification",
                f"node 'confirm' function 'verified' transitions to unknown node '{target}'",
            )

    def test_subflow_branch_targets(self):
        data = _composed()
        branch = {"field": "ok", "cases": {"true": "return", "false": "ask"}}
        data["flows"]["verification"]["nodes"]["confirm"] = _node(
            {"name": "check_again", "transition_to": branch}
        )
        FlowConfig.model_validate(data)

        branch["default"] = "book"
        self.assert_invalid(data, "transitions to unknown node 'book'")

    def test_subflow_cannot_have_a_node_named_return(self):
        data = _composed()
        data["flows"]["verification"]["nodes"]["return"] = _node()
        self.assert_invalid(data, "flows.verification", "cannot have a node named 'return'")

    def test_subflow_initial_node_must_exist(self):
        data = _composed()
        data["flows"]["verification"]["initial_node"] = "start"
        self.assert_invalid(
            data, "flows.verification", "initial_node 'start' is not a defined node"
        )

    def test_subflows_do_not_nest(self):
        data = _composed()
        data["flows"]["verification"]["nodes"]["inner"] = {"flow": "verification"}
        self.assert_invalid(
            data,
            "flows.verification",
            "node 'inner' runs a flow, which a subflow's nodes cannot do",
        )
        for key, value in (("flows", {}), ("global_functions", [])):
            data = _composed()
            data["flows"]["verification"][key] = value
            self.assert_invalid(data, f"flows.verification.{key}")

    def test_subflow_function_cannot_share_a_global_function_name(self):
        data = _composed()
        data["global_functions"] = [{"name": "check"}]
        self.assert_invalid(
            data, "flow 'verification' node 'ask' function 'check' is also a global function"
        )

    # Node names have no dots in a config that has subflows.

    def test_no_dots_in_node_names_with_subflows(self):
        data = _composed()
        data["nodes"]["book.slot"] = _node()
        self.assert_invalid(data, "node 'book.slot' has a '.' in its name")

        data = _composed()
        data["nodes"]["verify.caller"] = {"flow": "verification", "return_to": "book"}
        self.assert_invalid(data, "node 'verify.caller' has a '.' in its name")

        data = _composed()
        data["flows"]["verification"]["nodes"]["ask.again"] = _node()
        self.assert_invalid(data, "node 'ask.again' has a '.' in its name")

    def test_dots_are_allowed_without_subflows(self):
        data = _minimal()
        data["nodes"]["scheduling.find_slot"] = _node(_go("back", "a"))
        data["nodes"]["a"]["functions"] = [_go("find", "scheduling.find_slot")]
        FlowConfig.model_validate(data)

    # Every subflow is run.

    def test_unused_flow(self):
        data = _composed()
        data["flows"]["closing"] = {"initial_node": "bye", "nodes": {"bye": _node()}}
        self.assert_invalid(data, "flow 'closing' is not run by any node")

        data["nodes"]["goodbye"] = {"flow": "closing"}
        FlowConfig.model_validate(data)

    def test_one_flow_run_by_two_nodes(self):
        data = _composed()
        data["nodes"]["verify_again"] = {"flow": "verification", "return_to": "start"}
        FlowConfig.model_validate(data)
