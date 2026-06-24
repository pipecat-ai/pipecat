"""Tests for config_validator module (non-interactive mode)."""

import json

import pytest

from pipecat.cli.config_validator import (
    ConfigValidationError,
    config_to_json,
    load_config_from_file,
    validate_and_build_config,
)
from pipecat.cli.prompts.questions import ProjectConfig


class TestValidCascadeConfigs:
    """Test valid cascade mode configurations."""

    def test_minimal_web_cascade(self):
        config = validate_and_build_config(
            name="my-bot",
            bot_type="web",
            transport=["daily"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
        )
        assert isinstance(config, ProjectConfig)
        assert config.project_name == "my-bot"
        assert config.bot_type == "web"
        assert config.transports == ["daily"]
        assert config.mode == "cascade"
        assert config.stt_service == "deepgram_stt"
        assert config.llm_service == "openai_llm"
        assert config.tts_service == "cartesia_tts"
        assert config.realtime_service is None

    def test_multiple_transports(self):
        config = validate_and_build_config(
            name="my-bot",
            bot_type="web",
            transport=["daily", "smallwebrtc"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
        )
        assert config.transports == ["daily", "smallwebrtc"]

    def test_telephony_cascade(self):
        config = validate_and_build_config(
            name="call-bot",
            bot_type="telephony",
            transport=["twilio"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
        )
        assert config.bot_type == "telephony"
        assert config.transports == ["twilio"]

    def test_telephony_with_webrtc_backup(self):
        """Telephony bots can add WebRTC for local testing."""
        config = validate_and_build_config(
            name="call-bot",
            bot_type="telephony",
            transport=["telnyx", "smallwebrtc"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="elevenlabs_tts",
        )
        assert config.transports == ["telnyx", "smallwebrtc"]

    def test_bot_type_inferred_telephony(self):
        """Omitting --bot-type infers telephony from a telephony transport."""
        config = validate_and_build_config(
            name="call-bot",
            transport=["twilio"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
        )
        assert config.bot_type == "telephony"

    def test_bot_type_inferred_web(self):
        """Omitting --bot-type infers web when all transports are WebRTC."""
        config = validate_and_build_config(
            name="web-bot",
            transport=["smallwebrtc"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
        )
        assert config.bot_type == "web"

    def test_bot_type_inferred_telephony_wins_over_webrtc(self):
        """A telephony transport wins: twilio + smallwebrtc infers telephony."""
        config = validate_and_build_config(
            name="call-bot",
            transport=["twilio", "smallwebrtc"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
        )
        assert config.bot_type == "telephony"


class TestValidRealtimeConfigs:
    """Test valid realtime mode configurations."""

    def test_minimal_realtime(self):
        config = validate_and_build_config(
            name="rt-bot",
            bot_type="web",
            transport=["smallwebrtc"],
            mode="realtime",
            realtime="openai_realtime",
        )
        assert config.mode == "realtime"
        assert config.realtime_service == "openai_realtime"
        assert config.stt_service is None
        assert config.llm_service is None
        assert config.tts_service is None


class TestDefaults:
    """Test that defaults are applied correctly."""

    def test_deploy_to_cloud_default_true(self):
        config = validate_and_build_config(
            name="bot",
            bot_type="web",
            transport=["daily"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
        )
        assert config.deploy_to_cloud is True

    def test_video_output_forced_on_with_video_service(self):
        config = validate_and_build_config(
            name="bot",
            bot_type="web",
            transport=["daily"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
            video="tavus_video",
        )
        assert config.video_output is True


class TestTransportResolution:
    """Test transport name resolution."""

    def test_daily_pstn_dialin(self):
        config = validate_and_build_config(
            name="bot",
            bot_type="telephony",
            transport=["daily_pstn"],
            daily_pstn_mode="dial-in",
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
        )
        assert config.transports == ["daily_pstn_dialin"]
        assert config.daily_pstn_mode == "dial-in"

    def test_daily_pstn_dialout(self):
        config = validate_and_build_config(
            name="bot",
            bot_type="telephony",
            transport=["daily_pstn"],
            daily_pstn_mode="dial-out",
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
        )
        assert config.transports == ["daily_pstn_dialout"]
        assert config.daily_pstn_mode == "dial-out"

    def test_twilio_daily_sip_dialin(self):
        config = validate_and_build_config(
            name="bot",
            bot_type="telephony",
            transport=["twilio_daily_sip"],
            twilio_daily_sip_mode="dial-in",
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
        )
        assert config.transports == ["twilio_daily_sip_dialin"]
        assert config.twilio_daily_sip_mode == "dial-in"

    def test_twilio_daily_sip_dialout(self):
        config = validate_and_build_config(
            name="bot",
            bot_type="telephony",
            transport=["twilio_daily_sip"],
            twilio_daily_sip_mode="dial-out",
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
        )
        assert config.transports == ["twilio_daily_sip_dialout"]
        assert config.twilio_daily_sip_mode == "dial-out"


class TestMissingFields:
    """Test that missing required fields produce correct error messages."""

    def test_missing_all_required(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config()
        errors = exc_info.value.errors
        assert any("--name" in e for e in errors)
        assert any("--transport" in e for e in errors)
        assert any("--mode" in e for e in errors)
        # --bot-type is no longer required: it's inferred from the transports.
        assert not any("--bot-type is required" in e for e in errors)

    def test_missing_cascade_services(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="cascade",
            )
        errors = exc_info.value.errors
        assert any("--stt" in e for e in errors)
        assert any("--llm" in e for e in errors)
        assert any("--tts" in e for e in errors)

    def test_missing_realtime_service(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="realtime",
            )
        errors = exc_info.value.errors
        assert any("--realtime" in e for e in errors)

    def test_missing_daily_pstn_mode(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="telephony",
                transport=["daily_pstn"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
            )
        errors = exc_info.value.errors
        assert any("--daily-pstn-mode" in e for e in errors)

    def test_missing_twilio_daily_sip_mode(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="telephony",
                transport=["twilio_daily_sip"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
            )
        errors = exc_info.value.errors
        assert any("--twilio-daily-sip-mode" in e for e in errors)


class TestInvalidValues:
    """Test that invalid service values are rejected."""

    def test_invalid_bot_type(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="mobile",
                transport=["daily"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
            )
        assert any("bot-type" in e for e in exc_info.value.errors)

    def test_invalid_mode(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="hybrid",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
            )
        assert any("mode" in e for e in exc_info.value.errors)

    def test_invalid_transport(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["nonexistent"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
            )
        assert any("Unknown transport" in e for e in exc_info.value.errors)

    def test_invalid_stt_service(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="cascade",
                stt="invalid_stt",
                llm="openai_llm",
                tts="cartesia_tts",
            )
        assert any("Unknown STT" in e for e in exc_info.value.errors)

    def test_invalid_llm_service(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="cascade",
                stt="deepgram_stt",
                llm="invalid_llm",
                tts="cartesia_tts",
            )
        assert any("Unknown LLM" in e for e in exc_info.value.errors)

    def test_invalid_tts_service(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="invalid_tts",
            )
        assert any("Unknown TTS" in e for e in exc_info.value.errors)

    def test_invalid_realtime_service(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="realtime",
                realtime="invalid_realtime",
            )
        assert any("Unknown realtime" in e for e in exc_info.value.errors)

    def test_invalid_video_service(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
                video="invalid_video",
            )
        assert any("Unknown video" in e for e in exc_info.value.errors)


class TestCrossFieldConstraints:
    """Test cross-field validation constraints."""

    def test_video_only_for_web(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="telephony",
                transport=["twilio"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
                video="tavus_video",
            )
        assert any("web bots" in e.lower() for e in exc_info.value.errors)

    def test_krisp_requires_cloud(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
                deploy_to_cloud=False,
                enable_krisp=True,
            )
        assert any("krisp" in e.lower() for e in exc_info.value.errors)

    def test_video_input_only_for_web(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="telephony",
                transport=["twilio"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
                video_input=True,
            )
        assert any(
            "video-input" in e.lower() or "video input" in e.lower() for e in exc_info.value.errors
        )

    def test_telephony_transport_for_web_rejected(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["twilio"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
            )
        assert any("telephony transport" in e.lower() for e in exc_info.value.errors)

    def test_cascade_rejects_realtime_flag(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
                realtime="openai_realtime",
            )
        assert any(
            "realtime" in e.lower() and "cascade" in e.lower() for e in exc_info.value.errors
        )

    def test_realtime_rejects_stt_flag(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="realtime",
                realtime="openai_realtime",
                stt="deepgram_stt",
            )
        assert any("stt" in e.lower() and "realtime" in e.lower() for e in exc_info.value.errors)

    def test_client_framework_only_for_web(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="telephony",
                transport=["twilio"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
                client_framework="react",
            )
        assert any(
            "client-framework" in e.lower() or "web bots" in e.lower()
            for e in exc_info.value.errors
        )

    def test_client_server_requires_framework(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="cascade",
                stt="deepgram_stt",
                llm="openai_llm",
                tts="cartesia_tts",
                client_server="vite",
            )
        assert any(
            "client-server" in e.lower() and "client-framework" in e.lower()
            for e in exc_info.value.errors
        )

    def test_multiple_errors_collected(self):
        """All errors should be collected, not just the first."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_and_build_config(
                name="bot",
                bot_type="web",
                transport=["daily"],
                mode="cascade",
                # Missing all three cascade services
            )
        errors = exc_info.value.errors
        assert len(errors) == 3  # stt, llm, tts


class TestClientConfig:
    """Test client configuration."""

    def test_react_vite_client(self):
        config = validate_and_build_config(
            name="bot",
            bot_type="web",
            transport=["daily"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
            client_framework="react",
            client_server="vite",
        )
        assert config.generate_client is True
        assert config.client_framework == "react"
        assert config.client_server == "vite"

    def test_react_nextjs_client(self):
        config = validate_and_build_config(
            name="bot",
            bot_type="web",
            transport=["daily"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
            client_framework="react",
            client_server="nextjs",
        )
        assert config.generate_client is True
        assert config.client_server == "nextjs"

    def test_vanilla_client(self):
        config = validate_and_build_config(
            name="bot",
            bot_type="web",
            transport=["daily"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
            client_framework="vanilla",
        )
        assert config.generate_client is True
        assert config.client_framework == "vanilla"
        assert config.client_server == "vite"

    def test_no_client(self):
        config = validate_and_build_config(
            name="bot",
            bot_type="web",
            transport=["daily"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
            client_framework="none",
        )
        assert config.generate_client is False
        assert config.client_framework is None


class TestConfigFile:
    """Test config file loading."""

    def test_load_config_file(self, tmp_path):
        config_data = {
            "project_name": "file-bot",
            "bot_type": "web",
            "transports": ["daily"],
            "mode": "cascade",
            "stt_service": "deepgram_stt",
            "llm_service": "openai_llm",
            "tts_service": "cartesia_tts",
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        loaded = load_config_from_file(config_file)
        assert loaded["project_name"] == "file-bot"
        assert loaded["bot_type"] == "web"

    def test_load_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config_from_file(tmp_path / "nonexistent.json")

    def test_load_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            load_config_from_file(bad_file)


def _parse_config_dict(file_data: dict) -> ProjectConfig:
    """Simulate the merging logic from resolve_scaffold_config: map config dict keys to
    validate_and_build_config kwargs, exactly as the CLI does after loading JSON."""
    return validate_and_build_config(
        name=file_data.get("name") or file_data.get("project_name"),
        bot_type=file_data.get("bot_type"),
        transport=file_data.get("transports") or file_data.get("transport"),
        mode=file_data.get("mode"),
        stt=file_data.get("stt") or file_data.get("stt_service"),
        llm=file_data.get("llm") or file_data.get("llm_service"),
        tts=file_data.get("tts") or file_data.get("tts_service"),
        realtime=file_data.get("realtime") or file_data.get("realtime_service"),
        video=file_data.get("video") or file_data.get("video_service"),
        client_framework=file_data.get("client_framework"),
        client_server=file_data.get("client_server"),
        daily_pstn_mode=file_data.get("daily_pstn_mode"),
        twilio_daily_sip_mode=file_data.get("twilio_daily_sip_mode"),
        recording=file_data.get("recording", False),
        transcription=file_data.get("transcription", False),
        video_input=file_data.get("video_input", False),
        video_output=file_data.get("video_output", False),
        deploy_to_cloud=file_data.get("deploy_to_cloud", True),
        enable_krisp=file_data.get("enable_krisp", False),
        observability=file_data.get("observability", file_data.get("enable_observability", False)),
    )


class TestConfigDictParsing:
    """Test that config dicts (as from a JSON file) parse into correct ProjectConfigs."""

    def test_cascade_with_service_suffix_keys(self):
        """Config files use *_service keys (matching ProjectConfig field names)."""
        config = _parse_config_dict(
            {
                "project_name": "file-bot",
                "bot_type": "web",
                "transports": ["daily"],
                "mode": "cascade",
                "stt_service": "deepgram_stt",
                "llm_service": "openai_llm",
                "tts_service": "cartesia_tts",
            }
        )
        assert config.project_name == "file-bot"
        assert config.stt_service == "deepgram_stt"
        assert config.llm_service == "openai_llm"
        assert config.tts_service == "cartesia_tts"
        assert config.deploy_to_cloud is True

    def test_cascade_with_short_keys(self):
        """Config files can also use short keys (stt, llm, tts)."""
        config = _parse_config_dict(
            {
                "project_name": "short-bot",
                "bot_type": "web",
                "transports": ["smallwebrtc"],
                "mode": "cascade",
                "stt": "deepgram_stt",
                "llm": "anthropic_llm",
                "tts": "elevenlabs_tts",
            }
        )
        assert config.stt_service == "deepgram_stt"
        assert config.llm_service == "anthropic_llm"
        assert config.tts_service == "elevenlabs_tts"

    def test_realtime_config(self):
        config = _parse_config_dict(
            {
                "project_name": "rt-bot",
                "bot_type": "web",
                "transports": ["daily"],
                "mode": "realtime",
                "realtime_service": "openai_realtime",
            }
        )
        assert config.mode == "realtime"
        assert config.realtime_service == "openai_realtime"
        assert config.stt_service is None

    def test_telephony_with_daily_pstn(self):
        config = _parse_config_dict(
            {
                "project_name": "pstn-bot",
                "bot_type": "telephony",
                "transports": ["daily_pstn"],
                "daily_pstn_mode": "dial-in",
                "mode": "cascade",
                "stt_service": "deepgram_stt",
                "llm_service": "openai_llm",
                "tts_service": "cartesia_tts",
            }
        )
        assert config.transports == ["daily_pstn_dialin"]
        assert config.daily_pstn_mode == "dial-in"

    def test_full_featured_config(self):
        config = _parse_config_dict(
            {
                "project_name": "full-bot",
                "bot_type": "web",
                "transports": ["daily", "smallwebrtc"],
                "mode": "cascade",
                "stt_service": "deepgram_stt",
                "llm_service": "openai_llm",
                "tts_service": "cartesia_tts",
                "client_framework": "react",
                "client_server": "nextjs",
                "recording": True,
                "transcription": True,
                "video_input": True,
                "video_output": True,
                "deploy_to_cloud": False,
                "enable_observability": True,
            }
        )
        assert config.transports == ["daily", "smallwebrtc"]
        assert config.generate_client is True
        assert config.client_framework == "react"
        assert config.client_server == "nextjs"
        assert config.recording is True
        assert config.transcription is True
        assert config.video_input is True
        assert config.deploy_to_cloud is False
        assert config.enable_observability is True

    def test_invalid_config_dict_reports_errors(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            _parse_config_dict(
                {
                    "project_name": "bad-bot",
                    "bot_type": "web",
                    "transports": ["daily"],
                    "mode": "cascade",
                    # missing stt, llm, tts
                }
            )
        assert len(exc_info.value.errors) == 3

    def test_name_key_alias(self):
        """Both 'name' and 'project_name' should work."""
        config = _parse_config_dict(
            {
                "name": "alias-bot",
                "bot_type": "web",
                "transports": ["daily"],
                "mode": "cascade",
                "stt": "deepgram_stt",
                "llm": "openai_llm",
                "tts": "cartesia_tts",
            }
        )
        assert config.project_name == "alias-bot"


class TestConfigToJson:
    """Test JSON serialization of ProjectConfig."""

    def test_json_roundtrip(self):
        config = validate_and_build_config(
            name="json-bot",
            bot_type="web",
            transport=["daily"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
        )
        json_str = config_to_json(config)
        data = json.loads(json_str)
        assert data["project_name"] == "json-bot"
        assert data["bot_type"] == "web"
        assert data["transports"] == ["daily"]
        assert data["mode"] == "cascade"
        assert data["stt_service"] == "deepgram_stt"
        assert data["llm_service"] == "openai_llm"
        assert data["tts_service"] == "cartesia_tts"
        assert data["deploy_to_cloud"] is True

    def test_json_includes_all_fields(self):
        config = validate_and_build_config(
            name="full-bot",
            bot_type="web",
            transport=["daily"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
            recording=True,
            transcription=True,
            video_input=True,
            video_output=True,
            observability=True,
        )
        json_str = config_to_json(config)
        data = json.loads(json_str)
        assert data["recording"] is True
        assert data["transcription"] is True
        assert data["video_input"] is True
        assert data["video_output"] is True
        assert data["enable_observability"] is True


class TestFeatureFlags:
    """Test optional feature flags."""

    def test_all_features_enabled(self):
        config = validate_and_build_config(
            name="bot",
            bot_type="web",
            transport=["daily"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
            recording=True,
            transcription=True,
            video_input=True,
            video_output=True,
            observability=True,
            enable_krisp=True,
            deploy_to_cloud=True,
        )
        assert config.recording is True
        assert config.transcription is True
        assert config.video_input is True
        assert config.video_output is True
        assert config.enable_observability is True
        assert config.enable_krisp is True
        assert config.deploy_to_cloud is True

    def test_all_features_disabled(self):
        config = validate_and_build_config(
            name="bot",
            bot_type="web",
            transport=["daily"],
            mode="cascade",
            stt="deepgram_stt",
            llm="openai_llm",
            tts="cartesia_tts",
            recording=False,
            transcription=False,
            video_input=False,
            video_output=False,
            observability=False,
            enable_krisp=False,
            deploy_to_cloud=False,
        )
        assert config.recording is False
        assert config.transcription is False
        assert config.video_input is False
        assert config.video_output is False
        assert config.enable_observability is False
        assert config.enable_krisp is False
        assert config.deploy_to_cloud is False
