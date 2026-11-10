#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import unittest

from pipecat.extensions.deepgram_voice_agent.listen import DeepgramListenUpdater, create_listen
from pipecat.extensions.deepgram_voice_agent.processor import (
    DeepgramVoiceAgentProcessor,
)
from pipecat.extensions.deepgram_voice_agent.serializer import DeepgramVoiceAgentSerializer
from pipecat.extensions.deepgram_voice_agent.services import (
    SettingsError,
    check_endpoint_functions,
    function_schemas,
)
from pipecat.frames.frames import (
    ClientFunctionCallResultFrame,
    EndWorkerFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMMessagesAppendFrame,
    LLMUpdateSettingsFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.tests.utils import run_test


def _messages(frames) -> list[dict]:
    return [f.message for f in frames if isinstance(f, OutputTransportMessageUrgentFrame)]


class TestDeepgramVoiceAgentSerializer(unittest.IsolatedAsyncioTestCase):
    def _serializer(self, client_rate=16000) -> DeepgramVoiceAgentSerializer:
        return DeepgramVoiceAgentSerializer(
            client_in_sample_rate=client_rate,
            client_out_sample_rate=client_rate,
            pipeline_in_sample_rate=16000,
        )

    async def test_binary_is_audio(self):
        frame = await self._serializer().deserialize(b"\x01\x00\x02\x00")
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.audio, b"\x01\x00\x02\x00")
        self.assertEqual(frame.sample_rate, 16000)

    async def test_split_sample_is_kept_for_the_next_message(self):
        serializer = self._serializer()
        first = await serializer.deserialize(b"\x01\x00\x02")
        second = await serializer.deserialize(b"\x00")
        self.assertEqual(first.audio, b"\x01\x00")
        self.assertEqual(second.audio, b"\x02\x00")

    async def test_json_is_a_transport_message(self):
        frame = await self._serializer().deserialize('{"type": "KeepAlive"}')
        self.assertIsInstance(frame, InputTransportMessageFrame)
        self.assertEqual(frame.message, {"type": "KeepAlive"})

    async def test_text_that_is_not_json_is_passed_through(self):
        frame = await self._serializer().deserialize("not json")
        self.assertEqual(frame.message, "not json")

    async def test_outgoing_message_is_json(self):
        data = await self._serializer().serialize(
            OutputTransportMessageUrgentFrame(message={"type": "AgentAudioDone"})
        )
        self.assertEqual(json.loads(data), {"type": "AgentAudioDone"})

    async def test_outgoing_audio_is_raw_pcm(self):
        audio = b"\x01\x00" * 160
        data = await self._serializer().serialize(
            OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)
        )
        self.assertEqual(data, audio)


class TestDeepgramVoiceAgentProcessor(unittest.IsolatedAsyncioTestCase):
    async def _run(self, *messages, processor=None):
        processor = processor or DeepgramVoiceAgentProcessor(system_instruction="Be brief.")
        down, up = await run_test(
            processor,
            frames_to_send=[InputTransportMessageFrame(message=m) for m in messages],
        )
        return down, up

    async def test_function_call_response_becomes_a_result_frame(self):
        down, _ = await self._run({"type": "FunctionCallResponse", "id": "call_1", "content": "ok"})
        results = [f for f in down if isinstance(f, ClientFunctionCallResultFrame)]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool_call_id, "call_1")
        self.assertEqual(results[0].result, "ok")

    async def test_inject_user_message_runs_the_llm(self):
        down, _ = await self._run({"type": "InjectUserMessage", "content": "Hi"})
        appends = [f for f in down if isinstance(f, LLMMessagesAppendFrame)]
        self.assertEqual(appends[0].messages, [{"role": "user", "content": "Hi"}])
        self.assertTrue(appends[0].run_llm)
        self.assertIn(
            {"type": "ConversationText", "role": "user", "content": "Hi"}, _messages(down)
        )

    async def test_update_prompt_appends_to_the_system_instruction(self):
        down, _ = await self._run({"type": "UpdatePrompt", "prompt": "Be kind."})
        updates = [f for f in down if isinstance(f, LLMUpdateSettingsFrame)]
        self.assertEqual(updates[0].delta.system_instruction, "Be brief.\nBe kind.")
        self.assertIn({"type": "PromptUpdated"}, _messages(down))

    async def test_replace_prompt_replaces_the_system_instruction(self):
        down, _ = await self._run({"type": "ReplacePrompt", "prompt": "World state: ..."})
        updates = [f for f in down if isinstance(f, LLMUpdateSettingsFrame)]
        self.assertEqual(updates[0].delta.system_instruction, "World state: ...")

    async def test_update_speak_changes_the_voice(self):
        down, _ = await self._run(
            {"type": "UpdateSpeak", "speak": {"provider": {"model": "aura-2-zeus-en"}}}
        )
        updates = [f for f in down if isinstance(f, TTSUpdateSettingsFrame)]
        self.assertEqual(updates[0].delta.voice, "aura-2-zeus-en")
        self.assertIn({"type": "SpeakUpdated"}, _messages(down))

    async def test_inject_agent_message_is_spoken(self):
        down, _ = await self._run({"type": "InjectAgentMessage", "content": "Hello there"})
        self.assertEqual([f.text for f in down if isinstance(f, TTSSpeakFrame)], ["Hello there"])

    async def test_second_settings_ends_the_session(self):
        down, up = await self._run({"type": "Settings"})
        self.assertEqual(_messages(down)[0]["code"], "SETTINGS_ALREADY_APPLIED")
        self.assertTrue(any(isinstance(f, EndWorkerFrame) for f in up))

    async def test_unknown_message_ends_the_session(self):
        down, up = await self._run({"type": "ThisTypeDoesNotExist"})
        self.assertEqual(_messages(down)[0]["code"], "UNPARSABLE_CLIENT_MESSAGE")
        self.assertTrue(any(isinstance(f, EndWorkerFrame) for f in up))

    async def test_keep_alive_does_nothing(self):
        down, up = await self._run({"type": "KeepAlive"})
        self.assertEqual(_messages(down), [])
        self.assertEqual(up, [])


class TestDeepgramVoiceAgentServices(unittest.IsolatedAsyncioTestCase):
    def test_function_schemas(self):
        tools = function_schemas(
            [
                {
                    "name": "build",
                    "description": "Build something.",
                    "parameters": {
                        "type": "object",
                        "properties": {"site_id": {"type": "integer"}},
                        "required": ["site_id"],
                    },
                }
            ]
        )
        [schema] = tools.standard_tools
        self.assertEqual(schema.name, "build")
        self.assertEqual(schema.required, ["site_id"])

    async def test_non_https_endpoint_is_rejected(self):
        with self.assertRaises(SettingsError):
            await check_endpoint_functions(
                [{"name": "f", "endpoint": {"url": "http://example.com/f"}}]
            )

    async def test_private_endpoint_is_rejected(self):
        with self.assertRaises(SettingsError):
            await check_endpoint_functions(
                [{"name": "f", "endpoint": {"url": "https://127.0.0.1:2019/config"}}]
            )

    async def test_private_endpoint_is_allowed_when_configured(self):
        await check_endpoint_functions(
            [{"name": "f", "endpoint": {"url": "http://127.0.0.1:9999/f"}}],
            allow_nonpublic=True,
        )


if __name__ == "__main__":
    unittest.main()


class TestDeepgramVoiceAgentInjection(unittest.IsolatedAsyncioTestCase):
    async def _run(self, message, *, user_speaking=False, agent_speaking=False):
        processor = DeepgramVoiceAgentProcessor()
        processor._user_speaking = user_speaking
        processor._agent_speaking = agent_speaking
        down, _ = await run_test(processor, frames_to_send=[InputTransportMessageFrame(message)])
        return down

    async def test_agent_message_is_refused_while_the_user_speaks(self):
        down = await self._run(
            {"type": "InjectAgentMessage", "content": "Hi", "behavior": "queue"},
            user_speaking=True,
        )
        self.assertEqual(_messages(down)[0]["code"], "INJECT_AGENT_MESSAGE_DURING_USER_SPEECH")
        self.assertFalse(any(isinstance(f, TTSSpeakFrame) for f in down))

    async def test_agent_message_without_behavior_is_refused_while_the_agent_speaks(self):
        down = await self._run({"type": "InjectAgentMessage", "content": "Hi"}, agent_speaking=True)
        self.assertEqual(_messages(down)[0]["code"], "INJECT_AGENT_MESSAGE_DURING_AGENT_SPEECH")

    async def test_queued_agent_message_waits_for_the_agent(self):
        down = await self._run(
            {"type": "InjectAgentMessage", "content": "Hi", "behavior": "queue"},
            agent_speaking=True,
        )
        self.assertEqual([f.text for f in down if isinstance(f, TTSSpeakFrame)], ["Hi"])
        self.assertFalse(any(isinstance(f, InterruptionFrame) for f in down))

    async def test_interrupting_agent_message_barges_in(self):
        down = await self._run(
            {"type": "InjectAgentMessage", "content": "Hi", "behavior": "interrupt"},
            agent_speaking=True,
        )
        types = [type(f) for f in down]
        self.assertLess(types.index(InterruptionFrame), types.index(TTSSpeakFrame))

    async def test_injected_user_message_barges_in_and_ends_the_agent_turn(self):
        down = await self._run({"type": "InjectUserMessage", "content": "Hi"}, agent_speaking=True)
        self.assertTrue(any(isinstance(f, InterruptionFrame) for f in down))
        self.assertEqual(
            [m["type"] for m in _messages(down)],
            ["UserStartedSpeaking", "AgentAudioDone", "ConversationText", "History"],
        )


class TestDeepgramListenUpdater(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sent: list[dict] = []
        self.pushed: list = []

        async def send(message):
            self.sent.append(message)

        async def push_frame(frame):
            self.pushed.append(frame)

        self.listen = create_listen(
            {"provider": {"type": "deepgram", "model": "nova-3"}}, deepgram_api_key="test"
        )
        self.updater = DeepgramListenUpdater(
            self.listen,
            send=send,
            push_frame=push_frame,
            deepgram_api_key="test",
            http_session=None,  # type: ignore[arg-type]
        )

    async def _settle(self):
        await asyncio.sleep(0.05)

    async def test_starts_on_the_requested_api(self):
        self.assertIs(self.listen.active, self.listen.v1)

    async def test_unsupported_fields_are_refused_with_a_warning(self):
        await self.updater.update({"provider": {"model": "nova-3", "verbose": True}})
        self.assertEqual(self.sent[0]["type"], "Warning")
        self.assertEqual(self.sent[0]["code"], "UPDATE_LISTEN_UNSUPPORTED_FIELDS_CHANGED")
        self.assertIn("verbose", self.sent[0]["description"])

    async def test_too_many_keyterms_are_an_error(self):
        await self.updater.update({"provider": {"model": "nova-3", "keyterms": ["k"] * 101}})
        self.assertEqual(self.sent[0]["code"], "INVALID_SETTINGS")

    async def test_unchanged_settings_are_acknowledged(self):
        await self.updater.update({"provider": {"type": "deepgram", "model": "nova-3"}})
        await self._settle()
        self.assertEqual(self.sent, [{"type": "ListenUpdated"}])
        self.assertEqual(self.pushed, [])

    async def test_update_waits_for_the_user_to_stop_speaking(self):
        await self.updater.set_user_speaking(True)
        await self.updater.update({"provider": {"type": "deepgram", "model": "nova-3"}})
        await self._settle()
        self.assertEqual(self.sent, [])
        await self.updater.set_user_speaking(False)
        await self._settle()
        self.assertEqual(self.sent, [{"type": "ListenUpdated"}])
