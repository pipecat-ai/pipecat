#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import unittest

from pipecat.extensions.deepgram_voice_agent.listen import (
    DeepgramListenUpdater,
    check_listen_provider,
    create_listen,
)
from pipecat.extensions.deepgram_voice_agent.processor import (
    DeepgramVoiceAgentLLMInputGate,
    DeepgramVoiceAgentProcessor,
)
from pipecat.extensions.deepgram_voice_agent.serializer import DeepgramVoiceAgentSerializer
from pipecat.extensions.deepgram_voice_agent.services import (
    SettingsError,
    StageDirectionFilter,
    check_endpoint_functions,
    function_schemas,
    speak_settings,
)
from pipecat.frames.frames import (
    ClientFunctionCallResultFrame,
    EndWorkerFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    LLMUpdateSettingsFrame,
    ManuallySwitchServiceFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
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
        processor = DeepgramVoiceAgentProcessor()
        processor._agent_speaking = True
        down, _ = await run_test(
            processor,
            frames_to_send=[
                InputTransportMessageFrame(
                    {"type": "InjectAgentMessage", "content": "Hi", "behavior": "queue"}
                )
            ],
        )
        self.assertFalse(any(isinstance(f, (TTSSpeakFrame, InterruptionFrame)) for f in down))
        self.assertEqual(list(processor._queued_messages), ["Hi"])

    async def test_queued_agent_message_is_dropped_by_a_barge_in(self):
        processor = DeepgramVoiceAgentProcessor()
        processor._agent_speaking = True
        down, _ = await run_test(
            processor,
            frames_to_send=[
                InputTransportMessageFrame(
                    {"type": "InjectAgentMessage", "content": "Hi", "behavior": "queue"}
                ),
                InputTransportMessageFrame({"type": "InjectUserMessage", "content": "Stop"}),
            ],
        )
        self.assertEqual(list(processor._queued_messages), [])
        self.assertFalse(any(isinstance(f, TTSSpeakFrame) for f in down))

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


class TestDeepgramVoiceAgentAudioFormats(unittest.IsolatedAsyncioTestCase):
    def _serializer(self, **kwargs) -> DeepgramVoiceAgentSerializer:
        return DeepgramVoiceAgentSerializer(
            client_in_sample_rate=8000,
            client_out_sample_rate=kwargs.pop("rate", 8000),
            pipeline_in_sample_rate=16000,
            **kwargs,
        )

    def _audio(self, rate=8000) -> OutputAudioRawFrame:
        return OutputAudioRawFrame(
            audio=b"\x10\x00" * (rate // 10), sample_rate=rate, num_channels=1
        )

    async def test_mulaw_output_is_one_byte_per_sample(self):
        data = await self._serializer(output_encoding="mulaw").serialize(self._audio())
        self.assertEqual(len(data), 800)

    async def test_mulaw_input_becomes_pipeline_pcm(self):
        frame = await self._serializer(input_encoding="mulaw").deserialize(b"\xff" * 800)
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.sample_rate, 16000)

    async def test_wav_container_sends_a_header_once(self):
        serializer = self._serializer(output_container="wav")
        first = await serializer.serialize(self._audio())
        second = await serializer.serialize(self._audio())
        self.assertTrue(first.startswith(b"RIFF") and b"WAVEfmt " in first[:16])
        self.assertFalse(second.startswith(b"RIFF"))

    def test_unsupported_encoding_is_refused(self):
        with self.assertRaises(ValueError):
            self._serializer(output_encoding="mp3")


class TestDeepgramVoiceAgentValidation(unittest.IsolatedAsyncioTestCase):
    async def test_stage_directions_are_not_spoken(self):
        stage_filter = StageDirectionFilter()
        self.assertEqual(await stage_filter.filter("*beep*"), "")
        self.assertEqual(await stage_filter.filter("Hi *waves* there."), "Hi there.")

    def test_speak_speed_is_passed_on(self):
        settings = speak_settings({"provider": {"model": "aura-2-thalia-en", "speed": 1.1}})
        self.assertEqual(settings.voice, "aura-2-thalia-en")
        self.assertEqual(settings.speed, 1.1)

    def test_custom_speak_endpoint_is_refused(self):
        with self.assertRaises(SettingsError):
            speak_settings({"provider": {"model": "aura-2-thalia-en"}, "endpoint": {"url": "x"}})

    def test_language_hints_need_the_multilingual_model(self):
        with self.assertRaises(ValueError) as error:
            check_listen_provider({"language_hints": ["en"]}, model="flux-general-en")
        self.assertIn("flux-general-multi", str(error.exception))
        check_listen_provider({"language_hints": ["en"]}, model="flux-general-multi")

    def test_keyterms_with_surrounding_whitespace_are_refused(self):
        with self.assertRaises(ValueError):
            check_listen_provider({"keyterms": [" padded "]}, model="nova-3")


class TestDeepgramVoiceAgentLLMInputGate(unittest.IsolatedAsyncioTestCase):
    async def test_drops_one_inference_when_armed(self):
        gate = DeepgramVoiceAgentLLMInputGate()
        gate.skip_next_inference = True
        down, _ = await run_test(
            gate, frames_to_send=[LLMContextFrame(LLMContext()), LLMContextFrame(LLMContext())]
        )
        self.assertEqual(len([f for f in down if isinstance(f, LLMContextFrame)]), 1)

    async def test_a_new_turn_disarms_it(self):
        gate = DeepgramVoiceAgentLLMInputGate()
        gate.skip_next_inference = True
        down, _ = await run_test(
            gate, frames_to_send=[UserStartedSpeakingFrame(), LLMContextFrame(LLMContext())]
        )
        self.assertEqual(len([f for f in down if isinstance(f, LLMContextFrame)]), 1)


class TestDeepgramVoiceAgentSpeakFallback(unittest.IsolatedAsyncioTestCase):
    def _processor(self, services):
        processor = DeepgramVoiceAgentProcessor()
        processor.tts_services = services
        self.sent: list[dict] = []
        self.pushed: list = []
        self.failed = 0

        async def send(message):
            self.sent.append(message)

        async def push_frame(frame, direction=None):
            self.pushed.append(frame)

        async def on_speech_failed():
            self.failed += 1

        processor.send = send
        processor.speech_injector.push_frame = push_frame
        processor.on_speech_failed = on_speech_failed
        return processor

    async def test_failed_speech_is_retried_on_the_next_speak_entry(self):
        aura, flux = DeepgramVoiceAgentLLMInputGate(), DeepgramVoiceAgentLLMInputGate()
        processor = self._processor([aura, flux])
        processor.last_speech_request = "Hello there."
        await processor._on_speak_failure(aura, "HTTP 400")
        # The same failure reported again is handled once.
        await processor._on_speak_failure(aura, "completed with no audio")
        self.assertEqual([m["code"] for m in self.sent], ["SPEAK_REQUEST_FAILED"])
        self.assertEqual(self.failed, 1)
        switch, speak = self.pushed
        self.assertIsInstance(switch, ManuallySwitchServiceFrame)
        self.assertIs(switch.service, flux)
        self.assertEqual(speak.text, "Hello there.")

    async def test_last_speak_entry_failing_fails_to_speak(self):
        aura, flux = DeepgramVoiceAgentLLMInputGate(), DeepgramVoiceAgentLLMInputGate()
        processor = self._processor([aura, flux])
        processor.last_speech_request = "Hello there."
        await processor._on_speak_failure(aura, "HTTP 400")
        await processor._on_speak_failure(flux, "HTTP 503")
        self.assertEqual(
            [m["code"] for m in self.sent],
            ["SPEAK_REQUEST_FAILED", "SPEAK_REQUEST_FAILED", "FAILED_TO_SPEAK"],
        )
        self.assertEqual(self.failed, 2)
