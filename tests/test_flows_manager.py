#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Test suite for FlowManager functionality.

This module contains tests for the FlowManager class, which handles conversation
flow management across different LLM providers. Tests cover:
- Flow initialization
- State transitions and validation
- Function registration and execution
- Action handling
- Error cases

The tests use unittest.IsolatedAsyncioTestCase for async support and
include mocked dependencies for PipelineTask and LLM services.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from pipecat.flows.exceptions import FlowError, FlowTransitionError
from pipecat.flows.manager import FlowManager, NodeConfig
from pipecat.flows.types import FlowArgs, FlowResult, FlowsFunctionSchema, flows_tool_options
from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import LLMSettings
from tests.flows_test_helpers import (
    assert_tts_speak_frames_queued,
    get_advertised_tool_handlers,
    get_advertised_tools,
    make_mock_task,
)


class TestFlowManager(unittest.IsolatedAsyncioTestCase):
    """Test suite for FlowManager class.

    Tests functionality of FlowManager including:
    - Flow initialization
    - State transitions
    - Function registration
    - Action execution
    - Error handling
    - Node validation
    """

    async def asyncSetUp(self):
        """Set up test fixtures before each test."""
        self.mock_task = make_mock_task()
        self.mock_llm = OpenAILLMService(api_key="test-key")

        # Create mock assistant aggregator with public property only
        self.mock_assistant_aggregator = MagicMock()
        type(self.mock_assistant_aggregator).has_function_calls_in_progress = PropertyMock(
            return_value=False  # Default to no functions in progress
        )

        # Create mock context aggregator
        self.mock_context_aggregator = MagicMock()
        self.mock_context_aggregator.user = MagicMock()
        self.mock_context_aggregator.user.return_value = MagicMock()

        self.mock_context_aggregator.assistant = MagicMock(
            return_value=self.mock_assistant_aggregator
        )

        self.mock_result_callback = AsyncMock()

        # Sample node configurations
        self.sample_node: NodeConfig = {
            "role_message": "You are a helpful test assistant.",
            "task_messages": [{"role": "developer", "content": "Complete the test task."}],
            "functions": [
                FlowsFunctionSchema(
                    name="test_function",
                    description="Test function",
                    properties={},
                    required=[],
                    handler=AsyncMock(return_value={"status": "success"}),
                ),
            ],
        }

    async def test_worker_and_task_arguments(self):
        """Test the worker argument and the deprecated task argument."""
        # worker= is the canonical argument
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        self.assertIs(flow_manager.worker, self.mock_task)

        # task= still works but is deprecated
        with self.assertWarns(DeprecationWarning):
            flow_manager = FlowManager(
                task=self.mock_task,
                llm=self.mock_llm,
                context_aggregator=self.mock_context_aggregator,
            )
        self.assertIs(flow_manager.worker, self.mock_task)

        # The task property still resolves to the worker, but is deprecated
        with self.assertWarns(DeprecationWarning):
            self.assertIs(flow_manager.task, self.mock_task)

        # Passing both is an error
        with self.assertRaises(ValueError):
            FlowManager(
                worker=self.mock_task,
                task=self.mock_task,
                llm=self.mock_llm,
                context_aggregator=self.mock_context_aggregator,
            )

        # Passing neither is an error
        with self.assertRaises(ValueError):
            FlowManager(
                llm=self.mock_llm,
                context_aggregator=self.mock_context_aggregator,
            )

    async def test_flow_initialization(self):
        """Test initialization of flow."""
        # Create mock transition callback
        mock_function = AsyncMock()

        # Initialize flow manager
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )

        # Create test node
        test_node: NodeConfig = {
            "name": "test",
            "task_messages": [{"role": "developer", "content": "Test message"}],
            "functions": [
                FlowsFunctionSchema(
                    name="test_function",
                    description="Test function",
                    properties={},
                    required=[],
                    handler=mock_function,
                ),
            ],
        }

        # Initialize and set node
        await flow_manager.initialize()
        await flow_manager.set_node_from_config(test_node)

        self.assertFalse(mock_function.called)  # Shouldn't be called until function is used
        self.assertEqual(flow_manager._current_node, "test")

    async def test_node_validation(self):
        """Test node configuration validation."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Test missing task_messages
        invalid_config = {"functions": []}
        with self.assertRaises(FlowError) as context:
            await flow_manager.set_node_from_config(invalid_config)
        self.assertIn("missing required 'task_messages' field", str(context.exception))

        # Test valid config
        valid_config = {"name": "test", "task_messages": []}
        await flow_manager.set_node_from_config(valid_config)

        self.assertEqual(flow_manager._current_node, "test")
        self.assertEqual(flow_manager._current_functions, set())

    async def test_function_registration(self):
        """Test that a node's functions are advertised with a handler for auto-registration."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Reset mock to clear initialization calls
        self.mock_task.queue_frames.reset_mock()

        # Set node with function
        await flow_manager.set_node_from_config(self.sample_node)

        # The tool is advertised carrying its handler, which the LLM service
        # registers when it sees the advertised tools.
        handlers = get_advertised_tool_handlers(self.mock_task)
        self.assertEqual(set(handlers), {"test_function"})
        self.assertTrue(callable(handlers["test_function"]))

    async def test_action_execution(self):
        """Test execution of pre and post actions."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Create node config with actions
        node_with_actions: NodeConfig = {
            "role_message": self.sample_node["role_message"],
            "task_messages": self.sample_node["task_messages"],
            "functions": self.sample_node["functions"],
            "pre_actions": [{"type": "tts_say", "text": "Pre action"}],
            "post_actions": [{"type": "tts_say", "text": "Post action"}],
        }

        # Reset mock to clear initialization calls
        self.mock_task.queue_frame.reset_mock()

        # Set node with actions
        await flow_manager.set_node_from_config(node_with_actions)

        assert_tts_speak_frames_queued(self.mock_task, ["Pre action", "Post action"])

    async def test_error_handling(self):
        """Test error handling in flow manager.

        Verifies:
        1. Cannot set node before initialization
        2. Initialization fails properly when task queue fails
        3. Node setting fails when task queue fails
        """
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )

        # Test setting node before initialization
        with self.assertRaises(FlowTransitionError):
            await flow_manager.set_node_from_config(self.sample_node)

        # Initialize normally
        await flow_manager.initialize()
        self.assertTrue(flow_manager._initialized)

        # Test node setting error
        self.mock_task.queue_frames.side_effect = Exception("Queue error")
        with self.assertRaises(FlowError):
            await flow_manager.set_node_from_config(self.sample_node)

        # Verify flow manager remains initialized despite error
        self.assertTrue(flow_manager._initialized)

    async def test_state_management(self):
        """Test state management across nodes."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Set state data
        test_value = "test_value"
        flow_manager.state["test_key"] = test_value

        # Reset mock to clear initialization calls
        self.mock_task.queue_frames.reset_mock()

        # Verify state persists across node transitions
        await flow_manager.set_node_from_config(self.sample_node)
        self.assertEqual(flow_manager.state["test_key"], test_value)

    async def test_multiple_function_registration(self):
        """Test registration of multiple functions."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Create node config with multiple functions
        node_config: NodeConfig = {
            "task_messages": [{"role": "developer", "content": "Test"}],
            "functions": [
                FlowsFunctionSchema(
                    name=f"func_{i}",
                    description=f"Function {i}",
                    properties={},
                    required=[],
                    handler=AsyncMock(return_value={"status": "success"}),
                )
                for i in range(3)
            ],
        }

        await flow_manager.set_node_from_config(node_config)

        # Verify all functions were advertised (each carrying a handler) and tracked
        handlers = get_advertised_tool_handlers(self.mock_task)
        self.assertEqual(set(handlers), {"func_0", "func_1", "func_2"})
        self.assertEqual(len(flow_manager._current_functions), 3)

    async def test_advertised_handlers_register_with_node_call_options(self):
        """Advertised handlers register with each tool's resolved call options.

        The wrapped handler carries the tool's call options (via @tool_options),
        so the LLM service resolves cancel_on_interruption to Flows' default of
        False — not the service's own default of True — and honors explicit
        overrides on both FlowsFunctionSchemas and direct functions.
        """
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        async def handler(args, flow_manager):
            return {"ok": True}, None

        @flows_tool_options(cancel_on_interruption=True, timeout_secs=7)
        async def direct_tool(flow_manager, city: str):
            """Do a thing.

            Args:
                city: A city.
            """
            return {"ok": True}, None

        await flow_manager.set_node_from_config(
            {
                "task_messages": [{"role": "developer", "content": "Test"}],
                "functions": [
                    FlowsFunctionSchema(
                        name="defaults",
                        description="Uses default call options",
                        properties={},
                        required=[],
                        handler=handler,
                    ),
                    FlowsFunctionSchema(
                        name="overrides",
                        description="Overrides call options",
                        properties={},
                        required=[],
                        handler=handler,
                        cancel_on_interruption=True,
                        timeout_secs=12.5,
                    ),
                    direct_tool,
                ],
            }
        )

        # Register the advertised tools the way the LLM service does on inference.
        self.mock_llm._sync_registered_tool_handlers(get_advertised_tools(self.mock_task))

        # FlowsFunctionSchema default: Flows' False default survives (not the service's True).
        defaults = self.mock_llm._functions["defaults"]
        self.assertFalse(defaults.cancel_on_interruption)
        self.assertIsNone(defaults.timeout_secs)

        # FlowsFunctionSchema explicit overrides are honored.
        overrides = self.mock_llm._functions["overrides"]
        self.assertTrue(overrides.cancel_on_interruption)
        self.assertEqual(overrides.timeout_secs, 12.5)

        # A direct function's @flows_tool_options values are honored.
        direct = self.mock_llm._functions["direct_tool"]
        self.assertTrue(direct.cancel_on_interruption)
        self.assertEqual(direct.timeout_secs, 7)

    async def test_redeclared_function_rebinds_new_handler(self):
        """Regression: redeclaring a function in a new node must bind the new handler.

        Two adjacent nodes declare ``go`` with different handlers returning
        different next nodes. The handler advertised for ``go`` must reflect
        the latest node's handler, not the first one's.

        See https://github.com/pipecat-ai/pipecat-flows/issues/269.
        """
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        handler_a_calls = []
        handler_b_calls = []

        async def handler_a(args, flow_manager):
            handler_a_calls.append(args)
            return {"from": "A"}, {
                "task_messages": [{"role": "developer", "content": "menu"}],
                "functions": [],
            }

        async def handler_b(args, flow_manager):
            handler_b_calls.append(args)
            return {"from": "B"}, {
                "task_messages": [{"role": "developer", "content": "home"}],
                "functions": [],
            }

        await flow_manager.set_node_from_config(
            {
                "task_messages": [{"role": "developer", "content": "A"}],
                "functions": [
                    FlowsFunctionSchema(
                        name="go",
                        description="A's go",
                        properties={},
                        required=[],
                        handler=handler_a,
                    ),
                ],
            }
        )
        await flow_manager.set_node_from_config(
            {
                "task_messages": [{"role": "developer", "content": "B"}],
                "functions": [
                    FlowsFunctionSchema(
                        name="go",
                        description="B's go",
                        properties={},
                        required=[],
                        handler=handler_b,
                    ),
                ],
            }
        )

        # After node B, the advertised ``go`` handler must be node B's, not node A's.
        latest_go = get_advertised_tool_handlers(self.mock_task)["go"]

        async def result_callback(result, *, properties=None):
            pass

        params = FunctionCallParams(
            function_name="go",
            tool_call_id="t1",
            arguments={},
            llm=None,
            pipeline_worker=self.mock_task,
            context=None,
            result_callback=result_callback,
        )
        await latest_go(params)

        self.assertEqual(len(handler_b_calls), 1, "handler_b should have run")
        self.assertEqual(len(handler_a_calls), 0, "handler_a should NOT have run")

    async def test_initialize_already_initialized(self):
        """Test initializing an already initialized flow manager."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Try to initialize again
        with patch("loguru.logger.warning") as mock_logger:
            await flow_manager.initialize()
            mock_logger.assert_called_once()

    async def test_register_action(self):
        """Test registering custom actions."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )

        async def custom_action(action):
            pass

        flow_manager.register_action("custom", custom_action)
        self.assertIn("custom", flow_manager._action_manager._action_handlers)

    async def test_call_handler_variations(self):
        """Test different handler signature variations."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Test handler with args
        async def handler_with_args(args):
            return {"status": "success", "args": args}

        result = await flow_manager._call_handler(handler_with_args, {"test": "value"})
        self.assertEqual(result["args"]["test"], "value")

        # Test handler without args
        async def handler_no_args():
            return {"status": "success"}

        result = await flow_manager._call_handler(handler_no_args, {})
        self.assertEqual(result["status"], "success")

        # Test handler with FlowManager parameter (2+ parameters)
        async def handler_with_flow_manager(args, flow_manager_param):
            return {
                "status": "success",
                "has_flow_manager": True,
                "flow_manager": flow_manager_param,  # Return for verification
                "args": args,
            }

        result = await flow_manager._call_handler(handler_with_flow_manager, {"test": "value"})
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["has_flow_manager"])
        self.assertIs(result["flow_manager"], flow_manager)  # Verify it's the same instance
        self.assertTrue(isinstance(result["flow_manager"], FlowManager))
        self.assertEqual(result["args"]["test"], "value")

        # Test instance method handler
        class TestHandlerClass:
            def __init__(self):
                self.instance_data = "test_instance"

            async def instance_method_handler(self, args):
                return {"status": "success", "instance_data": self.instance_data, "args": args}

            async def instance_method_with_flow_manager(self, args, flow_manager_param):
                return {
                    "status": "success",
                    "has_flow_manager": True,
                    "flow_manager": flow_manager_param,  # Return for verification
                    "instance_data": self.instance_data,
                    "args": args,
                }

            @classmethod
            async def class_method_handler(cls, args):
                return {"status": "success", "class_data": "test_class", "args": args}

            @classmethod
            async def class_method_with_flow_manager(cls, args, flow_manager_param):
                return {
                    "status": "success",
                    "has_flow_manager": True,
                    "flow_manager": flow_manager_param,  # Return for verification
                    "class_data": "test_class",
                    "args": args,
                }

        test_instance = TestHandlerClass()

        # Test instance method (1 parameter after self)
        result = await flow_manager._call_handler(
            test_instance.instance_method_handler, {"test": "value"}
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["instance_data"], "test_instance")
        self.assertEqual(result["args"]["test"], "value")

        # Test instance method with FlowManager (2+ parameters after self)
        result = await flow_manager._call_handler(
            test_instance.instance_method_with_flow_manager, {"test": "value"}
        )
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["has_flow_manager"])
        self.assertIs(result["flow_manager"], flow_manager)  # Verify it's the same instance
        self.assertEqual(result["instance_data"], "test_instance")
        self.assertEqual(result["args"]["test"], "value")

        # Test classmethod (1 parameter after cls)
        result = await flow_manager._call_handler(
            TestHandlerClass.class_method_handler, {"test": "value"}
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["class_data"], "test_class")
        self.assertEqual(result["args"]["test"], "value")

        # Test classmethod with FlowManager (2+ parameters after cls)
        result = await flow_manager._call_handler(
            TestHandlerClass.class_method_with_flow_manager, {"test": "value"}
        )
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["has_flow_manager"])
        self.assertIs(result["flow_manager"], flow_manager)  # Verify it's the same instance
        self.assertEqual(result["class_data"], "test_class")
        self.assertEqual(result["args"]["test"], "value")

    async def test_transition_func_error_handling(self):
        """Test error handling in transition functions."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        async def error_handler(args):
            raise ValueError("Test error")

        transition_func = await flow_manager._create_transition_func("test", error_handler)

        # Mock result callback
        callback_called = False

        async def result_callback(result):
            nonlocal callback_called
            callback_called = True
            self.assertIn("error", result)
            self.assertEqual(result["status"], "error")
            self.assertIn("Test error", result["error"])

        # The transition function should catch the error and pass it to the callback
        params = FunctionCallParams(
            function_name="test",
            tool_call_id="id",
            arguments={},
            llm=None,
            pipeline_worker=self.mock_task,
            context=None,
            result_callback=result_callback,
        )
        await transition_func(params)
        self.assertTrue(callback_called, "Result callback was not called")

    async def test_node_validation_edge_cases(self):
        """Test edge cases in node validation."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Test invalid function format (dict instead of FlowsFunctionSchema)
        invalid_config = {
            "task_messages": [{"role": "developer", "content": "Test"}],
            "functions": [{"type": "function"}],
        }
        with self.assertRaises(FlowError) as context:
            await flow_manager.set_node_from_config(invalid_config)
        self.assertIn("Invalid function format", str(context.exception))

        # A FlowsFunctionSchema requires a handler: omitting it is a construction-time error.
        with self.assertRaises(TypeError):
            FlowsFunctionSchema(
                name="test_func",
                description="Test",
                properties={},
                required=[],
            )

    async def test_action_execution_error_handling(self):
        """Test error handling in action execution."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Create node config with actions that will fail
        node_config: NodeConfig = {
            "task_messages": [{"role": "developer", "content": "Test"}],
            "functions": [],
            "pre_actions": [{"type": "invalid_action"}],
            "post_actions": [{"type": "another_invalid_action"}],
        }

        # Should raise FlowError due to invalid actions
        with self.assertRaises(FlowError):
            await flow_manager.set_node_from_config(node_config)

        # Verify error handling for pre and post actions separately
        with self.assertRaises(FlowError):
            await flow_manager._execute_actions(pre_actions=[{"type": "invalid_action"}])

        with self.assertRaises(FlowError):
            await flow_manager._execute_actions(post_actions=[{"type": "invalid_action"}])

    async def test_update_llm_context_error_handling(self):
        """Test error handling in LLM context updates."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Mock worker to raise error on queue_frames
        flow_manager._worker.queue_frames.side_effect = Exception("Queue error")

        with self.assertRaises(FlowError):
            await flow_manager._update_llm_context(
                role_message=None,
                role_messages=None,
                task_messages=[{"role": "developer", "content": "Test"}],
                functions=[],
            )

    async def test_function_declarations_processing(self):
        """Test processing of function declarations format."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        async def test_handler(args):
            return {"status": "success"}

        # Create node config with multiple FlowsFunctionSchema functions
        node_config: NodeConfig = {
            "task_messages": [{"role": "developer", "content": "Test"}],
            "functions": [
                FlowsFunctionSchema(
                    name="test1",
                    description="Test function 1",
                    properties={},
                    required=[],
                    handler=test_handler,
                ),
                FlowsFunctionSchema(
                    name="test2",
                    description="Test function 2",
                    properties={},
                    required=[],
                    handler=test_handler,
                ),
            ],
        }

        # Set node and verify function registration
        await flow_manager.set_node_from_config(node_config)

        # Verify both functions were registered
        self.assertIn("test1", flow_manager._current_functions)
        self.assertIn("test2", flow_manager._current_functions)

    async def test_role_message_inheritance(self):
        """Test that role_message is sent as LLMUpdateSettingsFrame."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # First node with role_message (singular)
        first_node: NodeConfig = {
            "role_message": "You are a helpful assistant.",
            "task_messages": [{"role": "developer", "content": "First task."}],
            "functions": [],
        }

        # Second node without role messages
        second_node: NodeConfig = {
            "task_messages": [{"role": "developer", "content": "Second task."}],
            "functions": [],
        }

        # Set first node
        await flow_manager.set_node_from_config(first_node)
        first_call = self.mock_task.queue_frames.call_args_list[0]
        first_frames = first_call[0][0]

        # Verify LLMUpdateSettingsFrame with system_instruction
        settings_frames = [f for f in first_frames if isinstance(f, LLMUpdateSettingsFrame)]
        self.assertEqual(len(settings_frames), 1)
        self.assertEqual(
            settings_frames[0].delta.system_instruction, "You are a helpful assistant."
        )

        # Verify AppendFrame contains only task_messages (not role_messages)
        append_frames = [f for f in first_frames if isinstance(f, LLMMessagesAppendFrame)]
        self.assertEqual(len(append_frames), 1)
        self.assertEqual(append_frames[0].messages, first_node["task_messages"])

        # Verify frame ordering: LLMUpdateSettingsFrame before LLMMessagesAppendFrame
        settings_idx = first_frames.index(settings_frames[0])
        append_idx = first_frames.index(append_frames[0])
        self.assertLess(settings_idx, append_idx)

        # Reset mock and set second node
        self.mock_task.queue_frames.reset_mock()
        await flow_manager.set_node_from_config(second_node)

        # Verify no LLMUpdateSettingsFrame for second node (no role_messages)
        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]
        settings_frames = [f for f in second_frames if isinstance(f, LLMUpdateSettingsFrame)]
        self.assertEqual(len(settings_frames), 0)

        # Verify AppendFrame with only task messages
        append_frames = [f for f in second_frames if isinstance(f, LLMMessagesAppendFrame)]
        self.assertEqual(len(append_frames), 1)
        self.assertEqual(append_frames[0].messages, second_node["task_messages"])

    async def test_frame_type_selection(self):
        """Test that the context-update frame type follows the context strategy.

        Under the default (APPEND) strategy, the context update appends for
        every node.
        """
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        test_node: NodeConfig = {
            "task_messages": [{"role": "developer", "content": "Test task."}],
            "functions": [],
        }

        # Under the default strategy the first node appends.
        await flow_manager.set_node_from_config(test_node)
        first_call = self.mock_task.queue_frames.call_args_list[0]  # Get first call
        first_frames = first_call[0][0]
        self.assertTrue(
            any(isinstance(f, LLMMessagesAppendFrame) for f in first_frames),
            "First node should use AppendFrame under the default strategy",
        )
        self.assertFalse(
            any(isinstance(f, LLMMessagesUpdateFrame) for f in first_frames),
            "First node should not use UpdateFrame under the default strategy",
        )

        # Reset mock
        self.mock_task.queue_frames.reset_mock()

        # Subsequent node should also use AppendFrame
        await flow_manager.set_node_from_config(test_node)
        first_call = self.mock_task.queue_frames.call_args_list[0]  # Get first call
        second_frames = first_call[0][0]
        self.assertTrue(
            any(isinstance(f, LLMMessagesAppendFrame) for f in second_frames),
            "Subsequent nodes should use AppendFrame",
        )
        self.assertFalse(
            any(isinstance(f, LLMMessagesUpdateFrame) for f in second_frames),
            "Subsequent nodes should not use UpdateFrame",
        )

    async def test_edge_vs_node_function_behavior(self):
        """Test different completion behavior for edge and node functions."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Create test functions
        async def test_handler(args):
            return {"status": "success"}

        async def consolidated_test_handler_1(args):
            next_node = {
                "task_messages": [{"role": "developer", "content": "Next"}],
                "functions": [],
            }
            return {"status": "success"}, next_node

        async def consolidated_test_handler_2(args):
            next_node = {
                "task_messages": [{"role": "developer", "content": "Next"}],
                "functions": [],
            }
            return {"status": "success"}, next_node

        # Create node with both types of functions
        node_config: NodeConfig = {
            "name": "test",
            "task_messages": [{"role": "developer", "content": "Test"}],
            "functions": [
                FlowsFunctionSchema(
                    name="node_function",
                    description="Node function",
                    properties={},
                    required=[],
                    handler=test_handler,
                ),
                FlowsFunctionSchema(
                    name="edge_function_1",
                    description="Edge function",
                    properties={},
                    required=[],
                    handler=consolidated_test_handler_1,
                ),
                FlowsFunctionSchema(
                    name="edge_function_2",
                    description="Edge function",
                    properties={},
                    required=[],
                    handler=consolidated_test_handler_2,
                ),
            ],
        }

        await flow_manager.set_node_from_config(node_config)

        # Get the advertised handlers (which the LLM service auto-registers)
        handlers = get_advertised_tool_handlers(self.mock_task)
        node_func = handlers["node_function"]
        edge_func_1 = handlers["edge_function_1"]
        edge_func_2 = handlers["edge_function_2"]

        # Test node function
        self.mock_task.queue_frames.reset_mock()
        node_result = None
        node_properties = None

        async def node_callback(result, *, properties=None):
            nonlocal node_result, node_properties
            node_result = result
            node_properties = properties

        params_1 = FunctionCallParams(
            function_name="node_function",
            tool_call_id="id1",
            arguments={},
            llm=None,
            pipeline_worker=self.mock_task,
            context=None,
            result_callback=node_callback,
        )

        await node_func(params_1)
        # Node function should not set run_llm=False
        self.assertTrue(node_properties is None or node_properties.run_llm is not False)

        # Test edge function 1
        self.mock_task.queue_frames.reset_mock()
        edge_result_1 = None
        edge_properties_1 = None

        async def edge_callback_1(result, *, properties=None):
            nonlocal edge_result_1, edge_properties_1
            edge_result_1 = result
            edge_properties_1 = properties

        params_1 = FunctionCallParams(
            function_name="edge_function_1",
            tool_call_id="id2",
            arguments={},
            llm=None,
            pipeline_worker=self.mock_task,
            context=None,
            result_callback=edge_callback_1,
        )

        await edge_func_1(params_1)
        # Edge functions should set run_llm=False
        self.assertTrue(edge_properties_1 is not None and edge_properties_1.run_llm is False)

        # Test edge function 2
        self.mock_task.queue_frames.reset_mock()
        edge_result_2 = None
        edge_properties_2 = None

        async def edge_callback_2(result, *, properties=None):
            nonlocal edge_result_2, edge_properties_2
            edge_result_2 = result
            edge_properties_2 = properties

        params_2 = FunctionCallParams(
            function_name="edge_function_2",
            tool_call_id="id3",
            arguments={},
            llm=None,
            pipeline_worker=self.mock_task,
            context=None,
            result_callback=edge_callback_2,
        )

        await edge_func_2(params_2)
        # Edge functions should set run_llm=False
        self.assertTrue(edge_properties_2 is not None and edge_properties_2.run_llm is False)

    @patch("pipecat.flows.manager.LLMRunFrame")
    async def test_completion_timing(self, mock_llm_run_frame):
        """Test that completions occur at the right time."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Test initial node setup
        self.mock_task.queue_frames.reset_mock()
        mock_llm_run_frame.reset_mock()

        await flow_manager.set_node_from_config(
            {
                "task_messages": [{"role": "developer", "content": "Test"}],
                "functions": [],
            },
        )

        # Should see context update and completion trigger
        # First call is for updating context
        self.assertTrue(self.mock_task.queue_frames.called)

        # Verify that LLM completion was triggered by checking LLMRunFrame instantiation
        mock_llm_run_frame.assert_called_once()

        # Test node transition by directly setting next node
        next_node: NodeConfig = {
            "task_messages": [{"role": "developer", "content": "Next test"}],
            "functions": [],
        }

        self.mock_task.queue_frames.reset_mock()
        mock_llm_run_frame.reset_mock()

        await flow_manager.set_node_from_config(next_node)

        # Should see context update and completion trigger again
        self.assertTrue(self.mock_task.queue_frames.called)
        mock_llm_run_frame.assert_called_once()

    async def test_get_current_context(self):
        """Test getting current conversation context."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Mock the context messages
        mock_messages = [{"role": "developer", "content": "Test message"}]
        self.mock_context_aggregator.user()._context.get_messages.return_value = mock_messages

        # Test getting context
        context = flow_manager.get_current_context()
        self.assertEqual(context, mock_messages)

        # Test error when context aggregator is not available
        flow_manager._context_aggregator = None
        with self.assertRaises(FlowError) as context:
            flow_manager.get_current_context()
        self.assertIn("No context aggregator available", str(context.exception))

    async def test_handler_with_flow_manager(self):
        """Test function handler that receives both args and flow_manager."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        handler_called = False
        correct_flow_manager = False

        async def modern_handler(args: FlowArgs, flow_mgr: FlowManager) -> FlowResult:
            nonlocal handler_called, correct_flow_manager
            handler_called = True
            correct_flow_manager = flow_mgr is flow_manager
            return {"status": "success", "args_received": args, "has_flow_manager": True}

        result = await flow_manager._call_handler(modern_handler, {"test": "value"})

        self.assertTrue(handler_called)
        self.assertTrue(correct_flow_manager)
        self.assertEqual(result["args_received"]["test"], "value")
        self.assertTrue(result["has_flow_manager"])

    async def test_node_without_functions(self):
        """Test node configuration without functions field."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Create node config without functions field
        node_config: NodeConfig = {
            "task_messages": [{"role": "developer", "content": "Test task without functions."}],
        }

        # Set node and verify it works without error
        await flow_manager.set_node_from_config(node_config)

        # Verify current_functions is empty set
        self.assertEqual(flow_manager._current_functions, set())

        # Verify LLM tools were still set (with empty or placeholder functions)
        tools_frames_call = [
            call
            for call in self.mock_task.queue_frames.call_args_list
            if any(isinstance(frame, LLMSetToolsFrame) for frame in call[0][0])
        ]
        self.assertTrue(len(tools_frames_call) > 0, "Should have called LLMSetToolsFrame")

    async def test_node_with_empty_functions(self):
        """Test node configuration with empty functions list."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Create node config with empty functions list
        node_config: NodeConfig = {
            "task_messages": [{"role": "developer", "content": "Test task with empty functions."}],
            "functions": [],
        }

        # Set node and verify it works without error
        await flow_manager.set_node_from_config(node_config)

        # Verify current_functions is empty set
        self.assertEqual(flow_manager._current_functions, set())

        # Verify LLM tools were still set (with empty or placeholder functions)
        tools_frames_call = [
            call
            for call in self.mock_task.queue_frames.call_args_list
            if any(isinstance(frame, LLMSetToolsFrame) for frame in call[0][0])
        ]
        self.assertTrue(len(tools_frames_call) > 0, "Should have called LLMSetToolsFrame")

    async def test_role_message_singular(self):
        """Test that plain string role_message (singular) works correctly."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        node: NodeConfig = {
            "role_message": "You are a helpful assistant.",
            "task_messages": [{"role": "developer", "content": "Do the task."}],
            "functions": [],
        }

        await flow_manager.set_node_from_config(node)
        first_call = self.mock_task.queue_frames.call_args_list[0]
        first_frames = first_call[0][0]

        # Verify LLMUpdateSettingsFrame with correct system_instruction
        settings_frames = [f for f in first_frames if isinstance(f, LLMUpdateSettingsFrame)]
        self.assertEqual(len(settings_frames), 1)
        self.assertEqual(
            settings_frames[0].delta.system_instruction, "You are a helpful assistant."
        )

        # Verify messages frame contains only task_messages
        append_frames = [f for f in first_frames if isinstance(f, LLMMessagesAppendFrame)]
        self.assertEqual(len(append_frames), 1)
        self.assertEqual(append_frames[0].messages, node["task_messages"])

    async def test_role_messages_persist_across_reset(self):
        """Test that system instruction persists when a RESET node omits role_message."""
        from pipecat.flows.types import ContextStrategy, ContextStrategyConfig

        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.RESET),
        )
        await flow_manager.initialize()

        # First node sets role_message
        first_node: NodeConfig = {
            "role_message": "You are a helpful assistant.",
            "task_messages": [{"role": "developer", "content": "First task."}],
            "functions": [],
        }

        await flow_manager.set_node_from_config(first_node)
        first_call = self.mock_task.queue_frames.call_args_list[0]
        first_frames = first_call[0][0]

        # Verify first node sends LLMUpdateSettingsFrame
        settings_frames = [f for f in first_frames if isinstance(f, LLMUpdateSettingsFrame)]
        self.assertEqual(len(settings_frames), 1)
        self.assertEqual(
            settings_frames[0].delta.system_instruction, "You are a helpful assistant."
        )

        # Second node with RESET strategy but no role_messages
        self.mock_task.queue_frames.reset_mock()
        second_node: NodeConfig = {
            "task_messages": [{"role": "developer", "content": "Second task."}],
            "functions": [],
        }

        await flow_manager.set_node_from_config(second_node)
        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]

        # No LLMUpdateSettingsFrame since no role_message — system instruction
        # persists in LLM settings from the first node
        settings_frames = [f for f in second_frames if isinstance(f, LLMUpdateSettingsFrame)]
        self.assertEqual(len(settings_frames), 0)

        # Verify RESET still uses UpdateFrame for context messages
        update_frames = [f for f in second_frames if isinstance(f, LLMMessagesUpdateFrame)]
        self.assertEqual(len(update_frames), 1)
        self.assertEqual(update_frames[0].messages, second_node["task_messages"])

    async def test_role_messages_deprecated_warning(self):
        """Test that using role_messages (plural) emits a DeprecationWarning."""
        import warnings

        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        node: NodeConfig = {
            "role_messages": [{"role": "developer", "content": "You are a helpful assistant."}],
            "task_messages": [{"role": "developer", "content": "Do the task."}],
            "functions": [],
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await flow_manager.set_node_from_config(node)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertEqual(len(deprecation_warnings), 1)
            self.assertIn("role_messages", str(deprecation_warnings[0].message))
            self.assertIn("role_message", str(deprecation_warnings[0].message))

        # Verify the node still works correctly despite the warning —
        # legacy role_messages go into context messages, not LLMUpdateSettingsFrame
        first_call = self.mock_task.queue_frames.call_args_list[0]
        first_frames = first_call[0][0]
        settings_frames = [f for f in first_frames if isinstance(f, LLMUpdateSettingsFrame)]
        self.assertEqual(len(settings_frames), 0)

        append_frames = [f for f in first_frames if isinstance(f, LLMMessagesAppendFrame)]
        self.assertEqual(len(append_frames), 1)
        self.assertEqual(
            append_frames[0].messages[0],
            {"role": "developer", "content": "You are a helpful assistant."},
        )

        # Verify the warning is only emitted once
        self.mock_task.queue_frames.reset_mock()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await flow_manager.set_node_from_config(node)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertEqual(len(deprecation_warnings), 0)

    async def test_role_message_and_role_messages_both_specified(self):
        """Test that role_message takes precedence when both are specified."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        node: NodeConfig = {
            "role_message": "I am the preferred role.",
            "role_messages": [{"role": "developer", "content": "I am the deprecated role."}],
            "task_messages": [{"role": "developer", "content": "Do the task."}],
            "functions": [],
        }

        with patch("pipecat.flows.manager.logger") as mock_logger:
            await flow_manager.set_node_from_config(node)
            mock_logger.warning.assert_any_call(
                "Both 'role_message' and 'role_messages' specified; using 'role_message'"
            )

        first_call = self.mock_task.queue_frames.call_args_list[0]
        first_frames = first_call[0][0]
        settings_frames = [f for f in first_frames if isinstance(f, LLMUpdateSettingsFrame)]
        self.assertEqual(len(settings_frames), 1)
        self.assertEqual(settings_frames[0].delta.system_instruction, "I am the preferred role.")

    async def test_role_messages_list_format_still_works(self):
        """Test that legacy list-of-dicts role_messages are prepended to context messages."""
        import warnings

        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        node: NodeConfig = {
            "role_messages": [
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "developer", "content": "Be concise."},
            ],
            "task_messages": [{"role": "developer", "content": "Do the task."}],
            "functions": [],
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await flow_manager.set_node_from_config(node)
            # Should emit deprecation warning for role_messages
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertEqual(len(deprecation_warnings), 1)

        first_call = self.mock_task.queue_frames.call_args_list[0]
        first_frames = first_call[0][0]

        # Legacy role_messages should NOT produce LLMUpdateSettingsFrame
        settings_frames = [f for f in first_frames if isinstance(f, LLMUpdateSettingsFrame)]
        self.assertEqual(len(settings_frames), 0)

        # Legacy role_messages should be prepended to context messages
        append_frames = [f for f in first_frames if isinstance(f, LLMMessagesAppendFrame)]
        self.assertEqual(len(append_frames), 1)
        messages = append_frames[0].messages
        self.assertEqual(
            messages[0], {"role": "developer", "content": "You are a helpful assistant."}
        )
        self.assertEqual(messages[1], {"role": "developer", "content": "Be concise."})
        self.assertEqual(messages[2], {"role": "developer", "content": "Do the task."})
