#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pipecat.flows import FlowConfig
from pipecat.flows.validation import FlowReport, validate_flow

# --- Tools ---


async def choose_pizza(flow_manager):
    """User wants pizza."""
    return None, None


async def report_status(flow_manager, status: str):
    """Report a status.

    Args:
        status (str): The status.
    """
    return {"status": status}, None


async def finish(flow_manager):
    """Done."""
    return None, None


async def check_kitchen(action, flow_manager):
    """Pre-action handler."""


def not_async(flow_manager):
    """Not a coroutine."""


TOOLS = SimpleNamespace(
    choose_pizza=choose_pizza,
    report_status=report_status,
    finish=finish,
    check_kitchen=check_kitchen,
    not_async=not_async,
    NOT_A_TOOL="constant",
)

GOOD = """
initial_node: start
nodes:
  start:
    role_message: You work for {{ restaurant }}.
    task_messages:
      - role: developer
        content: Greet {{ caller }}.
    pre_actions:
      - type: function
        handler: check_kitchen
    functions:
      - name: choose_pizza
        transition_to: pizza
  pizza:
    task_messages:
      - role: developer
        content: Take the order.
    functions:
      - name: report_status
        transition_to:
          field: status
          cases:
            ok: end
            retry: pizza
  end:
    task_messages:
      - role: developer
        content: Bye, {{ caller }}.
    post_actions:
      - type: tts_say
        text: Thanks {{ caller }}!
      - type: end_conversation
global_functions:
  - name: finish
    transition_to: end
"""


def codes(report: FlowReport, level: str | None = None) -> list[str]:
    return [i.code for i in report.issues if level is None or i.level == level]


class TestLoading(unittest.TestCase):
    def test_good_config_from_text(self):
        report = validate_flow(GOOD)
        self.assertTrue(report.ok)
        self.assertEqual(report.issues, [])
        self.assertIsInstance(report.config, FlowConfig)

    def test_inventory(self):
        report = validate_flow(GOOD)
        self.assertEqual(report.tools, ["check_kitchen", "choose_pizza", "finish", "report_status"])
        self.assertEqual(report.variables, ["caller", "restaurant"])

    def test_from_path_and_mapping_and_config(self):
        d = Path(tempfile.mkdtemp())
        path = d / "flow.yaml"
        path.write_text(GOOD, encoding="utf-8")
        self.assertTrue(validate_flow(path).ok)
        config = FlowConfig.from_yaml(GOOD)
        self.assertTrue(validate_flow(config).ok)
        self.assertTrue(validate_flow(config.model_dump()).ok)

    def test_invalid_yaml(self):
        report = validate_flow("initial_node: [unclosed\nnodes: {")
        self.assertFalse(report.ok)
        self.assertEqual(codes(report), ["parse"])
        self.assertIsNone(report.config)

    def test_field_errors_are_all_reported(self):
        report = validate_flow(
            "initial_node: a\nnodes:\n  a:\n    task_messages: []\n    bogus: 1\n    other: 2\n"
        )
        self.assertFalse(report.ok)
        self.assertEqual(codes(report), ["schema", "schema"])
        self.assertIn("nodes.a.bogus", report.issues[0].message)
        self.assertIn("nodes.a.other", report.issues[1].message)

    def test_graph_errors_name_the_reference(self):
        report = validate_flow("initial_node: missing\nnodes:\n  a:\n    task_messages: []\n")
        self.assertEqual(codes(report), ["schema"])
        self.assertIn("initial_node 'missing'", report.issues[0].message)

    def test_top_level_not_mapping(self):
        report = validate_flow("- a list\n")
        self.assertEqual(codes(report), ["load"])

    def test_missing_file(self):
        report = validate_flow(Path(tempfile.mkdtemp()) / "nope.yaml")
        self.assertEqual(codes(report), ["load"])

    def test_to_dict(self):
        d = validate_flow(GOOD).to_dict()
        self.assertEqual(set(d), {"ok", "issues", "tools", "variables"})
        self.assertTrue(d["ok"])


class TestGraphWarnings(unittest.TestCase):
    def test_unreachable_node(self):
        report = validate_flow(
            "initial_node: a\nnodes:\n"
            "  a: {task_messages: [{role: developer, content: a}], post_actions: [{type: end_conversation}]}\n"
            "  orphan: {task_messages: [{role: developer, content: o}], post_actions: [{type: end_conversation}]}\n"
        )
        self.assertTrue(report.ok)
        self.assertEqual(codes(report), ["unreachable_node"])
        self.assertEqual(report.issues[0].node, "orphan")

    def test_global_function_makes_nodes_reachable(self):
        report = validate_flow(
            "initial_node: a\nnodes:\n"
            "  a: {task_messages: [{role: developer, content: a}]}\n"
            "  end: {task_messages: [{role: developer, content: e}], post_actions: [{type: end_conversation}]}\n"
            "global_functions: [{name: finish, transition_to: end}]\n"
        )
        self.assertEqual(report.issues, [])

    def test_dead_end(self):
        report = validate_flow(
            "initial_node: a\nnodes:\n"
            "  a: {task_messages: [{role: developer, content: a}], functions: [{name: choose_pizza, transition_to: b}]}\n"
            "  b: {task_messages: [{role: developer, content: b}], functions: [{name: report_status}]}\n"
        )
        self.assertEqual(codes(report), ["dead_end"])
        self.assertEqual(report.issues[0].node, "b")

    def test_self_loop_only_is_a_dead_end(self):
        report = validate_flow(
            "initial_node: a\nnodes:\n"
            "  a: {task_messages: [{role: developer, content: a}], functions: [{name: again, transition_to: a}]}\n"
        )
        self.assertEqual(codes(report), ["dead_end"])

    def test_end_node_is_not_a_dead_end(self):
        report = validate_flow(
            "initial_node: a\nnodes:\n"
            "  a: {task_messages: [{role: developer, content: a}], post_actions: [{type: end_conversation}]}\n"
        )
        self.assertEqual(report.issues, [])

    def test_branch_single_target(self):
        report = validate_flow(
            "initial_node: a\nnodes:\n"
            "  a:\n    task_messages: [{role: developer, content: a}]\n"
            "    functions:\n      - name: report_status\n        transition_to: {field: status, cases: {ok: b, meh: b}, default: b}\n"
            "  b: {task_messages: [{role: developer, content: b}], post_actions: [{type: end_conversation}]}\n"
        )
        self.assertEqual(codes(report), ["branch_single_target"])
        self.assertEqual(report.issues[0].function, "report_status")
        self.assertIn("every case and the default lead to 'b'", report.issues[0].message)

    def test_single_case_without_default_is_not_flagged(self):
        # One case plus the implicit "stay on the node" is two outcomes.
        report = validate_flow(
            "initial_node: a\nnodes:\n"
            "  a:\n    task_messages: [{role: developer, content: a}]\n"
            "    functions:\n      - name: report_status\n        transition_to: {field: status, cases: {ok: b}}\n"
            "  b: {task_messages: [{role: developer, content: b}], post_actions: [{type: end_conversation}]}\n"
        )
        self.assertEqual(report.issues, [])

    def test_warnings_do_not_affect_ok(self):
        report = validate_flow(
            "initial_node: a\nnodes:\n  a: {task_messages: [{role: developer, content: a}]}\n"
        )
        self.assertTrue(report.ok)
        self.assertEqual(codes(report, "warning"), ["dead_end"])


class TestReferences(unittest.TestCase):
    def test_all_good_with_tools_and_variables(self):
        report = validate_flow(GOOD, tools=TOOLS, variables={"restaurant": "L", "caller": "C"})
        self.assertTrue(report.ok)
        self.assertEqual(report.issues, [])

    def test_tools_without_variables_still_checks_tools(self):
        report = validate_flow(GOOD, tools=TOOLS)
        self.assertTrue(report.ok)

    def test_missing_and_invalid_tools_are_all_reported(self):
        text = GOOD.replace("name: choose_pizza", "name: nope").replace(
            "name: finish", "name: not_async"
        )
        report = validate_flow(text, tools=TOOLS)
        self.assertFalse(report.ok)
        errors = report.errors
        self.assertEqual([e.code for e in errors], ["missing_tool", "invalid_tool"])
        self.assertEqual(errors[0].node, "start")
        self.assertEqual(errors[0].function, "nope")
        self.assertIsNone(errors[1].node)
        self.assertIn("must be async", errors[1].message)

    def test_non_callable_tool(self):
        report = validate_flow(GOOD.replace("name: choose_pizza", "name: NOT_A_TOOL"), tools=TOOLS)
        self.assertEqual(codes(report, "error"), ["invalid_tool"])

    def test_missing_handler(self):
        report = validate_flow(GOOD.replace("handler: check_kitchen", "handler: nope"), tools=TOOLS)
        self.assertEqual(codes(report, "error"), ["missing_handler"])
        self.assertIn("pre_actions", report.issues[0].message)

    def test_tools_from_mapping(self):
        mapping = {k: v for k, v in vars(TOOLS).items()}
        self.assertTrue(validate_flow(GOOD, tools=mapping).ok)

    def test_missing_variables(self):
        report = validate_flow(GOOD, variables={"caller": "C"})
        self.assertEqual(codes(report, "error"), ["missing_variable"])
        self.assertIn("restaurant", report.issues[0].message)

    def test_variables_not_checked_when_not_given(self):
        self.assertTrue(validate_flow(GOOD).ok)


if __name__ == "__main__":
    unittest.main()
