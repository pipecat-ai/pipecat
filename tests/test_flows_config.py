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
        from_json = FlowConfig.model_validate(json.loads(json.dumps(from_yaml.model_dump())))
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

    def test_shipped_schema_file_is_current(self):
        # The package ships the schema for editors and flow builders. Regenerate
        # it with `uv run python scripts/flows/write_flow_config_schema.py`.
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
        data["nodes"]["a"]["functions"] = [{"name": "f", "description": "no schema in YAML"}]
        self.assert_invalid(data, "description")

    def test_function_action_requires_handler(self):
        data = _minimal()
        data["nodes"]["a"]["pre_actions"] = [{"type": "function"}]
        self.assert_invalid(data, "a 'function' action requires a 'handler' name")

    def test_non_function_action_rejects_handler(self):
        data = _minimal()
        data["nodes"]["a"]["post_actions"] = [{"type": "notify", "handler": "notify_slack"}]
        self.assert_invalid(data, "action type 'notify' does not take a 'handler'")

    def test_context_strategy_values(self):
        data = _minimal()
        data["nodes"]["a"]["context_strategy"] = "reset_with_summary"
        self.assert_invalid(data, "context_strategy")


if __name__ == "__main__":
    unittest.main()
