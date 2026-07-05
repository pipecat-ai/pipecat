#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SpeachesLLMService's Gemma role-alternation normalizer.

vLLM serving Gemma rejects any non-alternating message history with HTTP 400
("Conversation roles must alternate user/assistant/user/assistant/..."), and
that 400 used to disconnect a live voice call (run 159, workflow 11). These
tests feed deliberately malformed histories — tool-call + tool-result turns,
consecutive same-role turns, assistant-first openings — through
``normalize_for_gemma`` and assert the output strictly alternates and starts
with ``user``.

The normalizer is imported by file path so the test runs without the full
pipecat dependency tree (it depends only on stdlib).
"""

import importlib.util
import unittest
from pathlib import Path

# Load the module directly from its file so we don't pull in the openai client
# (and the rest of pipecat) just to test two pure functions.
_LLM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "pipecat"
    / "services"
    / "speaches"
    / "llm.py"
)


def _load_normalizer():
    # Stub the heavy imports the module does at import time so we can load it
    # in isolation. We only need the pure helpers.
    import sys
    import types

    # Minimal loguru stub.
    if "loguru" not in sys.modules:
        loguru_stub = types.ModuleType("loguru")
        loguru_stub.logger = types.SimpleNamespace(
            debug=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            info=lambda *a, **k: None,
        )
        sys.modules["loguru"] = loguru_stub

    # Stub the two pipecat imports the module makes at top level.
    for name, attrs in (
        ("pipecat.services.openai.base_llm", {"OpenAILLMSettings": object}),
        ("pipecat.services.openai.llm", {"OpenAILLMService": object}),
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            # Ensure parent packages resolve.
            parts = name.split(".")
            for i in range(1, len(parts)):
                pkg = ".".join(parts[:i])
                if pkg not in sys.modules:
                    sys.modules[pkg] = types.ModuleType(pkg)
            sys.modules[name] = mod

    spec = importlib.util.spec_from_file_location("_speaches_llm_under_test", _LLM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load_normalizer()
normalize_for_gemma = _mod.normalize_for_gemma
is_already_alternating = _mod._is_already_alternating


def _roles(messages):
    return [m["role"] for m in messages]


def _assert_strict_alternation(testcase, messages):
    """Assert messages start with optional system, then strictly alternate user/assistant."""
    roles = _roles(messages)
    if roles and roles[0] == "system":
        roles = roles[1:]
    testcase.assertTrue(roles, "expected at least one non-system turn")
    testcase.assertEqual(roles[0], "user", f"history must start with user, got {roles}")
    for i in range(1, len(roles)):
        testcase.assertNotEqual(
            roles[i],
            roles[i - 1],
            f"roles must alternate but {roles} has a repeat at index {i}",
        )
        testcase.assertIn(roles[i], ("user", "assistant"))


class TestGemmaAlternationNormalizer(unittest.TestCase):
    def test_run_159_tool_call_plus_consecutive_assistant(self):
        """The exact failure shape from run 159.

        A normal assistant prose turn, then the recovered tool call appended as
        assistant(tool_calls), then the tool-role result — three non-user turns
        in a row that vLLM/Gemma 400s on.
        """
        bad = [
            {"role": "system", "content": "You are Tzipi."},
            {"role": "user", "content": "כמה עולה הביטוח?"},
            {"role": "assistant", "content": "רגע אחד, אני בודקת."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "recovered_0",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "מחיר ביטוח סטודנט"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "המחיר הוא 49 שקל לחודש.",
                "tool_call_id": "recovered_0",
            },
            {"role": "user", "content": "תודה"},
        ]

        before = _roles(bad)
        self.assertFalse(
            is_already_alternating(bad),
            "run-159 history should be detected as non-alternating",
        )

        out = normalize_for_gemma(bad)
        after = _roles(out)

        print("\nRUN 159 role sequence:")
        print("  before:", before)
        print("  after :", after)

        _assert_strict_alternation(self, out)
        # System preserved up front.
        self.assertEqual(out[0]["role"], "system")
        # The tool result + tool_calls + prose all collapse into a single
        # assistant turn that still alternates with the surrounding user turns.
        self.assertEqual(after, ["system", "user", "assistant", "user"])
        # The web_search call is rendered inline (Gemma's pythonic shape) so the
        # history still reflects that a call happened.
        assistant_text = out[2]["content"]
        self.assertIn("web_search(", assistant_text)
        # The tool result text is folded in, not dropped.
        self.assertIn("49 שקל", assistant_text)

    def test_consecutive_user_turns_merge(self):
        bad = [
            {"role": "user", "content": "שלום"},
            {"role": "user", "content": "אתם פתוחים?"},
            {"role": "assistant", "content": "כן, שלום!"},
        ]
        out = normalize_for_gemma(bad)
        _assert_strict_alternation(self, out)
        self.assertEqual(_roles(out), ["user", "assistant"])
        self.assertEqual(out[0]["content"], "שלום\nאתם פתוחים?")

    def test_assistant_first_history_drops_leading_assistant(self):
        bad = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "שלום, מדברת ציפי"},
            {"role": "user", "content": "היי"},
            {"role": "assistant", "content": "איך אפשר לעזור?"},
        ]
        out = normalize_for_gemma(bad)
        _assert_strict_alternation(self, out)
        self.assertEqual(_roles(out), ["system", "user", "assistant"])

    def test_two_consecutive_tool_results(self):
        """Two tool results back-to-back still collapse cleanly."""
        bad = [
            {"role": "user", "content": "מה המחיר ומתי פתוח?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"q":"price"}'},
                    }
                ],
            },
            {"role": "tool", "content": "49 שקל", "tool_call_id": "t1"},
            {"role": "developer", "content": "פתוח 9-17"},
            {"role": "assistant", "content": "המחיר 49 שקל, פתוח 9 עד 17."},
        ]
        out = normalize_for_gemma(bad)
        _assert_strict_alternation(self, out)
        self.assertEqual(_roles(out), ["user", "assistant"])

    def test_already_alternating_is_preserved_semantically(self):
        good = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        self.assertTrue(is_already_alternating(good))
        out = normalize_for_gemma(good)
        _assert_strict_alternation(self, out)
        self.assertEqual(_roles(out), ["system", "user", "assistant", "user"])
        self.assertEqual([m["content"] for m in out[1:]], ["a", "b", "c"])

    def test_empty_and_singleton(self):
        self.assertEqual(normalize_for_gemma([]), [])
        out = normalize_for_gemma([{"role": "user", "content": "hi"}])
        self.assertEqual(_roles(out), ["user"])

    def test_list_content_parts(self):
        """Content given as OpenAI content-parts lists is flattened, not crashed."""
        bad = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "user", "content": [{"type": "text", "text": "world"}]},
            {"role": "assistant", "content": "hi"},
        ]
        out = normalize_for_gemma(bad)
        _assert_strict_alternation(self, out)
        self.assertEqual(out[0]["content"], "hello\nworld")

    def test_output_always_alternates_fuzz(self):
        """Any permutation of role-laden junk must come out strictly alternating."""
        import itertools

        samples = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a1"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "x", "type": "function", "function": {"name": "end_call", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "tr", "tool_call_id": "x"},
            {"role": "developer", "content": "dev"},
        ]
        for combo in itertools.permutations(samples, 4):
            out = normalize_for_gemma(list(combo))
            if not out:
                continue
            roles = _roles(out)
            body = roles[1:] if roles and roles[0] == "system" else roles
            if not body:
                continue
            self.assertEqual(body[0], "user", f"combo {[(m['role']) for m in combo]} -> {roles}")
            for i in range(1, len(body)):
                self.assertNotEqual(body[i], body[i - 1], f"repeat in {roles}")
                self.assertIn(body[i], ("user", "assistant"))


is_template_valid = _mod._is_template_valid
normalize_preserving_tools = _mod.normalize_preserving_tools


def _call(name="transition_to_2", call_id="c1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


class TestTemplateValidator(unittest.TestCase):
    """_is_template_valid mirrors the gemma3 pythonic template: tool messages
    and assistant-with-tool_calls turns are exempt from alternation."""

    def test_structured_tool_history_is_valid_as_is(self):
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [_call()]},
            {"role": "tool", "content": '{"status": "done"}', "tool_call_id": "c1"},
            {"role": "assistant", "content": "next question"},
            {"role": "user", "content": "answer"},
        ]
        self.assertTrue(is_template_valid(msgs))

    def test_tool_without_id_is_invalid(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [_call()]},
            {"role": "tool", "content": "r"},
        ]
        self.assertFalse(is_template_valid(msgs))

    def test_leading_plain_assistant_is_invalid(self):
        msgs = [
            {"role": "assistant", "content": "greeting"},
            {"role": "user", "content": "hi"},
        ]
        self.assertFalse(is_template_valid(msgs))

    def test_same_role_plain_pair_across_tools_is_invalid(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [_call()]},
            {"role": "tool", "content": "r", "tool_call_id": "c1"},
            {"role": "user", "content": "again"},
        ]
        self.assertFalse(is_template_valid(msgs))


class TestNormalizePreservingTools(unittest.TestCase):
    """The structure-preserving pass keeps tool_calls / tool results intact —
    flattening them into assistant text teaches the model to SPEAK the call
    syntax and invented result JSON (observed live: run 4, workflow 2)."""

    def test_greeting_dropped_tools_preserved(self):
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "greeting"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [_call()]},
            {"role": "tool", "content": '{"status": "done"}', "tool_call_id": "c1"},
            {"role": "assistant", "content": "next"},
        ]
        out = normalize_preserving_tools(msgs)
        self.assertTrue(is_template_valid(out))
        # tool_calls survived structurally — no [func()] text in any content
        calls = [m for m in out if m.get("tool_calls")]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool_calls"][0]["function"]["name"], "transition_to_2")
        self.assertEqual(calls[0]["content"], "")  # None coerced for the template
        tools = [m for m in out if m.get("role") == "tool"]
        self.assertEqual(len(tools), 1)
        self.assertNotIn("greeting", str(out))
        for m in out:
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                self.assertNotIn("transition_to_2", m["content"] or "")
                self.assertNotIn('{"status"', m["content"] or "")

    def test_user_pair_across_tools_gets_filler_assistant(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [_call()]},
            {"role": "tool", "content": "r", "tool_call_id": "c1"},
            {"role": "user", "content": "again"},
        ]
        out = normalize_preserving_tools(msgs)
        self.assertTrue(is_template_valid(out))
        self.assertEqual([m["content"] for m in out if m["role"] == "user"], ["hi", "again"])

    def test_adjacent_same_role_plain_turns_merge(self):
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
        ]
        out = normalize_preserving_tools(msgs)
        self.assertTrue(is_template_valid(out))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["content"], "u1\nu2")
        self.assertEqual(out[1]["content"], "a1\na2")

    def test_tool_result_without_id_folds_to_assistant_text(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "orphan result"},
        ]
        out = normalize_preserving_tools(msgs)
        self.assertTrue(is_template_valid(out))
        self.assertEqual(out[-1]["role"], "assistant")
        self.assertIn("orphan result", out[-1]["content"])

    def test_input_never_mutated(self):
        msgs = [
            {"role": "assistant", "content": "greeting"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [_call()]},
        ]
        import copy

        snapshot = copy.deepcopy(msgs)
        normalize_preserving_tools(msgs)
        self.assertEqual(msgs, snapshot)

    def test_fuzz_output_always_valid_or_flattenable(self):
        import itertools

        samples = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": None, "tool_calls": [_call()]},
            {"role": "tool", "content": "tr", "tool_call_id": "c1"},
            {"role": "tool", "content": "orphan"},
            {"role": "developer", "content": "dev"},
        ]
        for combo in itertools.permutations(samples, 4):
            out = normalize_preserving_tools(list(combo))
            if not is_template_valid(out):
                # The service falls back to the legacy flattener — that output
                # must always be template-valid.
                flat = normalize_for_gemma(list(combo))
                self.assertTrue(
                    is_template_valid(flat),
                    f"flatten fallback invalid for {[(m['role']) for m in combo]}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
