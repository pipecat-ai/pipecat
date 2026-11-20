#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import MagicMock, PropertyMock

from pipecat.flows import FlowError, FlowManager, FlowsFunctionSchema, NodeConfig, announce_nodes
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from tests.flows_test_helpers import get_advertised_tool_handlers, make_mock_worker


class TestAnnounceNodes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_worker = make_mock_worker()
        self.llm = OpenAILLMService(api_key="test-key")
        assistant = MagicMock()
        type(assistant).has_function_calls_in_progress = PropertyMock(return_value=False)
        self.context_aggregator = MagicMock()
        self.context_aggregator.user = MagicMock(return_value=MagicMock())
        self.context_aggregator.assistant = MagicMock(return_value=assistant)

    def make_manager(self, worker=None) -> FlowManager:
        return FlowManager(
            worker=worker or self.mock_worker,
            llm=self.llm,
            context_aggregator=self.context_aggregator,
        )

    def sent_messages(self) -> list[dict]:
        return [call.args[0] for call in self.mock_worker.rtvi.send_server_message.call_args_list]

    async def invoke(self, name: str):
        handler = get_advertised_tool_handlers(self.mock_worker)[name]

        async def result_callback(result, *, properties=None):
            if properties is not None and properties.on_context_updated is not None:
                await properties.on_context_updated()

        await handler(
            FunctionCallParams(
                function_name=name,
                tool_call_id="t1",
                arguments={},
                llm=self.llm,
                pipeline_worker=self.mock_worker,
                context=None,
                result_callback=result_callback,
            )
        )

    async def test_nothing_is_sent_unless_asked(self):
        flow_manager = self.make_manager()
        await flow_manager.initialize(
            {"name": "start", "task_messages": [{"role": "developer", "content": "hi"}]}
        )
        self.assertEqual(self.sent_messages(), [])

    async def test_announces_each_node(self):
        find_slot: NodeConfig = {
            "name": "find_slot",
            "metadata": {"step": 2},
            "task_messages": [{"role": "developer", "content": "Offer times."}],
        }

        async def begin(args, flow_manager):
            return None, find_slot

        start: NodeConfig = {
            "name": "start",
            "task_messages": [{"role": "developer", "content": "Greet."}],
            "functions": [
                FlowsFunctionSchema(
                    name="begin",
                    description="Begin.",
                    properties={},
                    required=[],
                    handler=begin,
                    metadata={"exit": "begun"},
                )
            ],
        }
        flow_manager = self.make_manager()
        announce_nodes(flow_manager)

        await flow_manager.initialize(start)
        self.assertEqual(
            self.sent_messages(), [{"type": "flows", "event": "node", "node": "start"}]
        )

        await self.invoke("begin")
        self.assertEqual(
            self.sent_messages()[1],
            {
                "type": "flows",
                "event": "node",
                "node": "find_slot",
                "previous_node": "start",
                "function": "begin",
                "metadata": {"step": 2},
                "function_metadata": {"exit": "begun"},
            },
        )

    async def test_needs_rtvi(self):
        worker = PipelineWorker(Pipeline([]), enable_rtvi=False)
        flow_manager = self.make_manager(worker)
        with self.assertRaises(FlowError) as cm:
            announce_nodes(flow_manager)
        self.assertIn("RTVI enabled", str(cm.exception))

    async def test_rtvi_enabled_worker(self):
        worker = PipelineWorker(Pipeline([]))
        announce_nodes(self.make_manager(worker))
