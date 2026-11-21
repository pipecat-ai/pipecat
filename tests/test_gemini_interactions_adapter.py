#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import io
import json
import unittest

from loguru import logger

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.adapters.llm.services.gemini_interactions_adapter import (
    GeminiInteractionsLLMAdapter,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage

SIGNATURE = b"\x12\x34signature"
SIGNATURE_B64 = base64.b64encode(SIGNATURE).decode()
PLACEHOLDER_B64 = base64.b64encode(b"skip_thought_signature_validator").decode()

WEATHER = FunctionSchema(
    name="get_weather",
    description="Get the weather for a city.",
    properties={"city": {"type": "string"}},
    required=["city"],
)


def tool_call(call_id: str, city: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": city})},
    }


def tool_result(call_id: str, result: dict) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)}


def signature(bookmark: dict, value: bytes | str = SIGNATURE) -> LLMSpecificMessage:
    """A thought signature stored the way the Gemini services store them."""
    return LLMSpecificMessage(
        llm="google",
        message={"type": "thought_signature", "signature": value, "bookmark": bookmark},
    )


class TestGeminiInteractionsInvocationParams(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.adapter = GeminiInteractionsLLMAdapter()

    async def params(self, messages: list, **kwargs) -> dict:
        context = LLMContext(messages=messages, **kwargs)
        return dict(await self.adapter.get_llm_invocation_params(context))

    async def test_text_conversation(self):
        """Test user and assistant text becoming user_input and model_output steps."""
        params = await self.params(
            [
                {"role": "user", "content": "Name a color."},
                {"role": "assistant", "content": "Blue."},
                {"role": "user", "content": "Another?"},
            ]
        )

        self.assertIsNone(params["system_instruction"])
        self.assertEqual(
            params["input"],
            [
                {"type": "user_input", "content": [{"type": "text", "text": "Name a color."}]},
                {"type": "model_output", "content": [{"type": "text", "text": "Blue."}]},
                {"type": "user_input", "content": [{"type": "text", "text": "Another?"}]},
            ],
        )

    async def test_system_instruction_setting(self):
        """Test a service's system instruction applying when the context has none."""
        context = LLMContext(messages=[{"role": "user", "content": "Hi"}])
        params = await self.adapter.get_llm_invocation_params(
            context, system_instruction="Be kind."
        )
        self.assertEqual(params["system_instruction"], "Be kind.")

    async def test_input_ends_on_the_users_side(self):
        """Test a context ending with an assistant message gaining a trailing user turn."""
        params = await self.params(
            [
                {"role": "user", "content": "Count to three."},
                {"role": "assistant", "content": "One, two,"},
            ]
        )
        self.assertEqual(
            params["input"][-1],
            {"type": "user_input", "content": [{"type": "text", "text": "."}]},
        )

    async def test_signed_function_call(self):
        """Test a stored signature becoming a thought step ahead of its call."""
        params = await self.params(
            [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "tool_calls": [tool_call("call_1", "Paris")]},
                signature({"function_call": "call_1"}),
                tool_result("call_1", {"temperature_f": 72}),
            ]
        )

        self.assertEqual(
            params["input"][1:],
            [
                {"type": "thought", "signature": SIGNATURE_B64},
                {
                    "type": "function_call",
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": {"city": "Paris"},
                },
                {
                    "type": "function_result",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "result": {"temperature_f": 72},
                },
            ],
        )

    async def test_string_signature_passes_through(self):
        """Test a signature already in base64 text form being sent as it is."""
        params = await self.params(
            [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "tool_calls": [tool_call("call_1", "Paris")]},
                signature({"function_call": "call_1"}, value="already-base64"),
                tool_result("call_1", {"temperature_f": 72}),
            ]
        )
        self.assertEqual(params["input"][1], {"type": "thought", "signature": "already-base64"})

    async def test_unsigned_function_call_gets_placeholder(self):
        """Test a call with no signature, as from another LLM, getting the placeholder."""
        params = await self.params(
            [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "tool_calls": [tool_call("call_1", "Paris")]},
                tool_result("call_1", {"temperature_f": 72}),
            ]
        )
        self.assertEqual(params["input"][1], {"type": "thought", "signature": PLACEHOLDER_B64})
        self.assertEqual(params["input"][2]["type"], "function_call")

    async def test_parallel_calls_share_one_thought(self):
        """Test parallel calls signed on the first call becoming one thought, then the calls."""
        params = await self.params(
            [
                {"role": "user", "content": "Weather in Paris and Tokyo?"},
                {"role": "assistant", "tool_calls": [tool_call("call_1", "Paris")]},
                {"role": "assistant", "tool_calls": [tool_call("call_2", "Tokyo")]},
                signature({"function_call": "call_1"}),
                tool_result("call_1", {"temperature_f": 72}),
                tool_result("call_2", {"temperature_f": 60}),
            ]
        )

        self.assertEqual(
            [(s["type"], s.get("id") or s.get("call_id")) for s in params["input"][1:]],
            [
                ("thought", None),
                ("function_call", "call_1"),
                ("function_call", "call_2"),
                ("function_result", "call_1"),
                ("function_result", "call_2"),
            ],
        )

    async def test_interleaved_parallel_calls_regrouped(self):
        """Test a batch stored call-then-result per call being sent as calls, then results."""
        params = await self.params(
            [
                {"role": "user", "content": "Weather in Paris and Tokyo?"},
                {"role": "assistant", "tool_calls": [tool_call("call_1", "Paris")]},
                tool_result("call_1", {"temperature_f": 72}),
                {"role": "assistant", "tool_calls": [tool_call("call_2", "Tokyo")]},
                tool_result("call_2", {"temperature_f": 60}),
                signature({"function_call": "call_1"}),
                {"role": "user", "content": "And Rome?"},
                {"role": "assistant", "tool_calls": [tool_call("call_3", "Rome")]},
                signature({"function_call": "call_3"}),
                tool_result("call_3", {"temperature_f": 80}),
            ]
        )

        self.assertEqual(
            [(s["type"], s.get("id") or s.get("call_id")) for s in params["input"]],
            [
                ("user_input", None),
                ("thought", None),
                ("function_call", "call_1"),
                ("function_call", "call_2"),
                ("function_result", "call_1"),
                ("function_result", "call_2"),
                ("user_input", None),
                ("thought", None),
                ("function_call", "call_3"),
                ("function_result", "call_3"),
            ],
        )

    async def test_sequential_signed_calls_stay_apart(self):
        """Test a call signed on its own starting a new turn rather than joining the last batch."""
        params = await self.params(
            [
                {"role": "user", "content": "Weather in Paris, then Tokyo?"},
                {"role": "assistant", "tool_calls": [tool_call("call_1", "Paris")]},
                tool_result("call_1", {"temperature_f": 72}),
                {"role": "assistant", "tool_calls": [tool_call("call_2", "Tokyo")]},
                tool_result("call_2", {"temperature_f": 60}),
                signature({"function_call": "call_1"}),
                signature({"function_call": "call_2"}),
            ]
        )
        self.assertEqual(
            [s["type"] for s in params["input"]],
            [
                "user_input",
                "thought",
                "function_call",
                "function_result",
                "thought",
                "function_call",
                "function_result",
            ],
        )

    async def test_model_turn_starts_with_its_thought(self):
        """Test text stored apart from its signed calls still opening the turn with the thought."""
        params = await self.params(
            [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": "Let me check."},
                {"role": "assistant", "tool_calls": [tool_call("call_1", "Paris")]},
                signature({"function_call": "call_1"}),
                tool_result("call_1", {"temperature_f": 72}),
            ]
        )

        self.assertEqual(
            [s["type"] for s in params["input"]],
            ["user_input", "thought", "model_output", "function_call", "function_result"],
        )

    async def test_signed_text_turn(self):
        """Test a text reply's signature becoming a thought step ahead of the text."""
        params = await self.params(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                signature({"text": "Hello!"}),
                {"role": "user", "content": "Name a color."},
            ]
        )
        self.assertEqual(
            [s["type"] for s in params["input"]],
            ["user_input", "thought", "model_output", "user_input"],
        )

    async def test_image_input(self):
        """Test an inline image becoming an image content item with base64 data."""
        data = b"\x89PNG fake"
        url = f"data:image/png;base64,{base64.b64encode(data).decode()}"
        params = await self.params(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ]
        )

        self.assertEqual(
            params["input"],
            [
                {
                    "type": "user_input",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image",
                            "mime_type": "image/png",
                            "data": base64.b64encode(data).decode(),
                        },
                    ],
                }
            ],
        )

    async def test_non_image_media_types(self):
        """Test audio and documents getting their own content types."""
        audio = base64.b64encode(b"RIFF").decode()
        pdf = f"data:application/pdf;base64,{base64.b64encode(b'%PDF').decode()}"
        params = await self.params(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
                        {
                            "type": "file_base64",
                            "file": {
                                "file_data": pdf,
                                "filename": "a.pdf",
                                "mime_type": "application/pdf",
                            },
                        },
                    ],
                }
            ]
        )
        self.assertEqual(
            [(item["type"], item["mime_type"]) for item in params["input"][0]["content"]],
            [("audio", "audio/wav"), ("document", "application/pdf")],
        )

    async def test_tools(self):
        """Test standard tools becoming flat function tools."""
        params = await self.params(
            [{"role": "user", "content": "Hi"}], tools=ToolsSchema(standard_tools=[WEATHER])
        )
        self.assertEqual(
            params["tools"],
            [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        )

    async def test_builtin_tools_included(self):
        """Test tools a service injects into the adapter being sent with the context's."""
        self.adapter.builtin_tools["get_weather"] = WEATHER
        params = await self.params([{"role": "user", "content": "Hi"}])
        self.assertEqual([t["name"] for t in params["tools"]], ["get_weather"])

    async def test_system_instruction_sent_once(self):
        """Test a context with no plain-text message not repeating the system instruction."""
        url = f"data:image/png;base64,{base64.b64encode(b'png').decode()}"
        context = LLMContext(
            messages=[
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
            ]
        )
        params = await self.adapter.get_llm_invocation_params(
            context, system_instruction="Be kind."
        )
        self.assertEqual(params["system_instruction"], "Be kind.")
        self.assertEqual([s["type"] for s in params["input"]], ["user_input"])

    async def test_one_placeholder_per_unsigned_turn(self):
        """Test unsigned parallel calls opening their turn with a single placeholder."""
        params = await self.params(
            [
                {"role": "user", "content": "Weather in Rome and Oslo?"},
                {"role": "assistant", "tool_calls": [tool_call("call_1", "Rome")]},
                {"role": "assistant", "tool_calls": [tool_call("call_2", "Oslo")]},
                tool_result("call_1", {"temperature_f": 70}),
                tool_result("call_2", {"temperature_f": 50}),
            ]
        )
        self.assertEqual(
            [s["type"] for s in params["input"]],
            [
                "user_input",
                "thought",
                "function_call",
                "function_call",
                "function_result",
                "function_result",
            ],
        )
        self.assertEqual(params["input"][1]["signature"], PLACEHOLDER_B64)

    async def test_assistant_text_and_calls_in_one_message(self):
        """Test an assistant message carrying both text and calls keeping both."""
        params = await self.params(
            [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [tool_call("call_1", "Paris")],
                },
                signature({"function_call": "call_1"}),
                tool_result("call_1", {"temperature_f": 72}),
            ]
        )
        self.assertEqual(
            [s["type"] for s in params["input"]],
            ["user_input", "thought", "model_output", "function_call", "function_result"],
        )

    async def test_text_function_result(self):
        """Test a tool result that isn't a JSON object being sent as text."""
        params = await self.params(
            [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "tool_calls": [tool_call("call_1", "Paris")]},
                {"role": "tool", "tool_call_id": "call_1", "content": "COMPLETED"},
            ]
        )
        self.assertEqual(params["input"][-1]["result"], "COMPLETED")
        self.assertEqual(params["input"][-1]["name"], "get_weather")

    async def test_files_api_uri_passes_through(self):
        """Test a Gemini Files API URL being sent as a URI rather than inlined."""
        url = "https://generativelanguage.googleapis.com/v1beta/files/abc"
        params = await self.params(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file_url",
                            "file": {"url": url, "mime_type": "application/pdf"},
                        }
                    ],
                }
            ]
        )
        self.assertEqual(
            params["input"][0]["content"],
            [{"type": "document", "mime_type": "application/pdf", "uri": url}],
        )

    async def test_specific_step_passes_through(self):
        """Test an Interactions step stored as a Google-specific message being sent as it is."""
        step = {"type": "user_input", "content": [{"type": "text", "text": "Hi"}]}
        params = await self.params([LLMSpecificMessage(llm="google", message=step)])
        self.assertEqual(params["input"], [step])

    async def test_unsupported_specific_message_skipped(self):
        """Test a Google-specific message that isn't a step being left out, with a warning."""
        sink = io.StringIO()
        handler_id = logger.add(sink, level="WARNING", format="{message}")
        try:
            params = await self.params(
                [
                    LLMSpecificMessage(llm="google", message={"role": "user", "parts": []}),
                    {"role": "user", "content": "Hi"},
                ]
            )
        finally:
            logger.remove(handler_id)

        self.assertEqual([s["type"] for s in params["input"]], ["user_input"])
        self.assertIn("Skipping Google-specific", sink.getvalue())


class TestGeminiInteractionsTools(unittest.TestCase):
    def test_custom_gemini_tools(self):
        """Test built-in Gemini tools keyed by name becoming typed tools."""
        tools = ToolsSchema(
            standard_tools=[],
            custom_tools={AdapterType.GEMINI: [{"google_search": {}}, {"url_context": {}}]},
        )
        self.assertEqual(
            GeminiInteractionsLLMAdapter().to_provider_tools_format(tools),
            [{"type": "google_search"}, {"type": "url_context"}],
        )

    def test_typed_tools_pass_through(self):
        """Test tools already in the Interactions format passing through unchanged."""
        tool = {"type": "google_search", "search_types": ["web_search"]}
        self.assertEqual(GeminiInteractionsLLMAdapter.to_interactions_tools([tool]), [tool])

    def test_function_declarations(self):
        """Test each declaration of a function_declarations entry becoming its own tool."""
        tools = [{"function_declarations": [{"name": "a"}, {"name": "b"}]}]
        self.assertEqual(
            GeminiInteractionsLLMAdapter.to_interactions_tools(tools),
            [{"type": "function", "name": "a"}, {"type": "function", "name": "b"}],
        )


class TestGeminiInteractionsLogging(unittest.TestCase):
    def test_media_and_signatures_redacted(self):
        """Test the logging view hiding media data and thought signatures."""
        url = f"data:image/png;base64,{base64.b64encode(b'png').decode()}"
        context = LLMContext(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look"},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                },
                {"role": "assistant", "tool_calls": [tool_call("call_1", "Paris")]},
                signature({"function_call": "call_1"}),
                tool_result("call_1", {"temperature_f": 72}),
            ]
        )

        steps = GeminiInteractionsLLMAdapter().get_messages_for_logging(context)

        self.assertEqual(steps[0]["content"][1]["data"], "...")
        self.assertEqual(steps[1], {"type": "thought", "signature": "..."})
        # The context itself is untouched.
        self.assertEqual(context.messages[0]["content"][1]["image_url"]["url"], url)


if __name__ == "__main__":
    unittest.main()
