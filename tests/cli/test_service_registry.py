"""Tests for service registry integrity and completeness."""

import pytest

from pipecat.cli.registry import ServiceLoader, ServiceRegistry


class TestServiceRegistryIntegrity:
    """Test that the service registry is complete and consistent."""

    def test_no_duplicate_service_values(self):
        """Verify no duplicate service values exist."""
        all_services = []
        all_services.extend(ServiceRegistry.WEBRTC_TRANSPORTS)
        all_services.extend(ServiceRegistry.TELEPHONY_TRANSPORTS)
        all_services.extend(ServiceRegistry.STT_SERVICES)
        all_services.extend(ServiceRegistry.LLM_SERVICES)
        all_services.extend(ServiceRegistry.TTS_SERVICES)
        all_services.extend(ServiceRegistry.REALTIME_SERVICES)
        all_services.extend(ServiceRegistry.VIDEO_SERVICES)

        values = [s.value for s in all_services]
        duplicates = [v for v in values if values.count(v) > 1]

        if duplicates:
            pytest.fail(f"Duplicate service values found: {', '.join(set(duplicates))}")

    def test_package_extras_are_extractable(self):
        """Verify all service packages can have their extras extracted."""
        from pipecat.cli.registry import extract_package_extra

        all_services = []
        all_services.extend(ServiceRegistry.STT_SERVICES)
        all_services.extend(ServiceRegistry.LLM_SERVICES)
        all_services.extend(ServiceRegistry.TTS_SERVICES)
        all_services.extend(ServiceRegistry.REALTIME_SERVICES)
        all_services.extend(ServiceRegistry.VIDEO_SERVICES)

        for service in all_services:
            package = service.package
            # Should not raise an exception
            extras = extract_package_extra(package)
            # If package has brackets, extras should be extracted
            if "[" in package:
                assert len(extras) > 0, f"Failed to extract extras from {package}"


class TestServiceLoader:
    """Test ServiceLoader functionality."""

    def test_get_service_by_value(self):
        """Test finding a service by value."""
        service = ServiceLoader.get_service_by_value(ServiceRegistry.STT_SERVICES, "deepgram_stt")
        assert service is not None
        assert service.value == "deepgram_stt"
        assert service.label == "Deepgram"

    def test_get_service_by_value_not_found(self):
        """Test finding a non-existent service."""
        service = ServiceLoader.get_service_by_value(
            ServiceRegistry.STT_SERVICES, "nonexistent_service"
        )
        assert service is None

    def test_get_service_config(self):
        """Test retrieving service configuration."""
        config = ServiceLoader.get_service_config("deepgram_stt")
        assert config is not None
        assert "DeepgramSTTService" in config
        assert "DEEPGRAM_API_KEY" in config

    def test_nvidia_sagemaker_configs_use_aws_region(self):
        """Test NVIDIA SageMaker services use shared AWS_REGION."""
        stt_config = ServiceLoader.get_service_config("nvidia_sagemaker_stt")
        tts_config = ServiceLoader.get_service_config("nvidia_sagemaker_tts")

        assert stt_config is not None
        assert tts_config is not None
        assert 'endpoint_name=os.getenv("NVIDIA_SAGEMAKER_STT_ENDPOINT_NAME")' in stt_config
        assert 'endpoint_name=os.getenv("NVIDIA_SAGEMAKER_TTS_ENDPOINT_NAME")' in tts_config
        assert 'region=os.getenv("AWS_REGION")' in stt_config
        assert 'region=os.getenv("AWS_REGION")' in tts_config
        assert "NVIDIA_SAGEMAKER_STT_REGION" not in stt_config
        assert "NVIDIA_SAGEMAKER_TTS_REGION" not in tts_config
        assert 'voice=os.getenv("NVIDIA_SAGEMAKER_TTS_VOICE_ID")' in tts_config

    def test_get_service_import(self):
        """Test retrieving service imports."""
        imports = ServiceLoader.get_service_import("deepgram_stt")
        assert imports is not None
        assert len(imports) > 0
        assert any("DeepgramSTTService" in imp for imp in imports)

    def test_websocket_is_web_transport(self):
        """WebSocket is a web (WebRTC list) transport, not telephony."""
        web_options = ServiceLoader.get_transport_options("web")
        assert any(t.value == "websocket" for t in web_options)
        telephony_options = ServiceLoader.get_transport_options("telephony")
        assert all(t.value != "websocket" for t in telephony_options)

        transport = ServiceLoader.get_service_by_value(
            ServiceRegistry.WEBRTC_TRANSPORTS, "websocket"
        )
        assert transport is not None
        assert transport.package == "pipecat-ai[websocket]"

    def test_websocket_transport_imports(self):
        """WebSocket transport imports the params + Protobuf serializer.

        Bots build the transport via create_transport(), so only the params class
        (and the serializer the factory sets) are imported — not the transport class
        or the runner-args type.
        """
        imports = ServiceLoader.get_service_import("websocket")
        assert imports is not None
        joined = "\n".join(imports)
        assert "FastAPIWebsocketParams" in joined
        assert "ProtobufFrameSerializer" in joined

    @pytest.mark.parametrize(
        "service",
        ServiceRegistry.STT_SERVICES
        + ServiceRegistry.LLM_SERVICES
        + ServiceRegistry.TTS_SERVICES
        + ServiceRegistry.REALTIME_SERVICES
        + ServiceRegistry.VIDEO_SERVICES,
        ids=lambda s: s.value,
    )
    def test_every_service_package_is_extractable(self, service):
        """Test that every service's package extra can be extracted."""
        from pipecat.cli.registry import extract_package_extra

        service_value = service.value
        package = service.package

        # Should not raise an exception
        extras = extract_package_extra(package)

        # Package should have correct format
        assert package.startswith("pipecat-ai"), (
            f"Service {service_value} package should start with 'pipecat-ai', got: {package}"
        )

        # If package has brackets, extras should be extracted
        if "[" in package:
            assert len(extras) > 0, (
                f"Failed to extract extras from {service_value} package: {package}"
            )

    def test_extract_extras_for_cascade(self):
        """Test extracting extras for a cascade pipeline."""
        services = {
            "transports": ["daily"],
            "stt": "deepgram_stt",
            "llm": "openai_llm",
            "tts": "cartesia_tts",
        }

        extras = ServiceLoader.extract_extras_for_services(services)

        # Should always include these
        assert "runner" in extras
        assert "silero" in extras

        # Should include service-specific extras
        assert "daily" in extras
        assert "deepgram" in extras
        assert "openai" in extras
        assert "cartesia" in extras

    def test_extract_extras_for_realtime(self):
        """Test extracting extras for a realtime pipeline."""
        services = {
            "transports": ["daily"],
            "realtime": "openai_realtime",
        }

        extras = ServiceLoader.extract_extras_for_services(services)

        assert "runner" in extras
        assert "silero" in extras
        assert "daily" in extras
        assert "openai" in extras

    def test_extract_extras_with_video_service(self):
        """Test extracting extras when a video service is included."""
        services = {
            "transports": ["daily"],
            "stt": "deepgram_stt",
            "llm": "openai_llm",
            "tts": "cartesia_tts",
            "video": "tavus_video",
        }

        extras = ServiceLoader.extract_extras_for_services(services)

        # Should include video service extra
        assert "tavus" in extras

        # Should still include other service extras
        assert "runner" in extras
        assert "silero" in extras
        assert "daily" in extras
        assert "deepgram" in extras
        assert "openai" in extras
        assert "cartesia" in extras

    def test_extract_multi_extra_package(self):
        """Test extracting multiple extras from a multi-extra package string."""
        from pipecat.cli.registry import extract_package_extra

        extras = extract_package_extra("pipecat-ai[deepgram,sagemaker]")
        assert extras == ["deepgram", "sagemaker"]

    def test_extract_single_extra_package(self):
        """Test extracting a single extra returns a one-element list."""
        from pipecat.cli.registry import extract_package_extra

        extras = extract_package_extra("pipecat-ai[deepgram]")
        assert extras == ["deepgram"]

    def test_extract_no_extra_package(self):
        """Test extracting extras from a package with no extras returns empty list."""
        from pipecat.cli.registry import extract_package_extra

        extras = extract_package_extra("pipecat-ai")
        assert extras == []

    def test_extract_extras_for_sagemaker_service(self):
        """Test that SageMaker services produce separate extras, not a combined string."""
        # Find a SageMaker STT service if it exists
        sagemaker_stt = ServiceLoader.get_service_by_value(
            ServiceRegistry.STT_SERVICES, "deepgram_sagemaker_stt"
        )
        if sagemaker_stt is None:
            pytest.skip("deepgram_sagemaker_stt service not in registry")

        services = {
            "transports": ["daily"],
            "stt": "deepgram_sagemaker_stt",
            "llm": "openai_llm",
            "tts": "cartesia_tts",
        }

        extras = ServiceLoader.extract_extras_for_services(services)

        # Should contain separate "deepgram" and "sagemaker" extras, not "deepgram,sagemaker"
        assert "deepgram" in extras
        assert "sagemaker" in extras
        assert "deepgram,sagemaker" not in extras

    def test_validate_service_exists(self):
        """Test service existence validation."""
        assert ServiceLoader.validate_service_exists("deepgram_stt") is True
        assert ServiceLoader.validate_service_exists("daily") is True
        assert ServiceLoader.validate_service_exists("nonexistent") is False

    def test_observability_feature_imports_exist(self):
        """Test that observability feature imports are defined."""
        assert "observability" in ServiceRegistry.FEATURE_IMPORTS
        observability_imports = ServiceRegistry.FEATURE_IMPORTS["observability"]
        assert any("WhiskerObserver" in imp for imp in observability_imports)

    def test_get_imports_with_observability(self):
        """Test that observability imports are included when enabled."""
        services = {
            "transports": ["daily"],
            "stt": "deepgram_stt",
            "llm": "openai_llm",
            "tts": "cartesia_tts",
        }
        features = {
            "observability": True,
        }

        imports = ServiceLoader.get_imports_for_services(services, features, "web")

        # Check that observability imports are included
        import_str = "\n".join(imports)
        assert "WhiskerObserver" in import_str
        assert "pipecat_whisker" in import_str

    def test_get_imports_with_video_service(self):
        """Test that video service imports are included when a video service is selected."""
        services = {
            "transports": ["daily"],
            "stt": "deepgram_stt",
            "llm": "openai_llm",
            "tts": "cartesia_tts",
            "video": "tavus_video",
        }
        features = {
            "observability": False,
        }

        imports = ServiceLoader.get_imports_for_services(services, features, "web")

        # Check that video service imports are included
        import_str = "\n".join(imports)
        assert "TavusVideoService" in import_str
        assert "pipecat.services.tavus" in import_str

    def test_video_services_have_correct_metadata(self):
        """Test that all video services have the expected metadata."""
        for video_service in ServiceRegistry.VIDEO_SERVICES:
            # All video services should have env_prefix
            assert video_service.env_prefix is not None, f"{video_service.value} missing env_prefix"

            # All video services should have class_name
            assert video_service.class_name is not None, f"{video_service.value} missing class_name"
            assert len(video_service.class_name) > 0, f"{video_service.value} has empty class_name"

            # All video services should have include_params
            assert video_service.include_params is not None, (
                f"{video_service.value} missing include_params"
            )
            assert "api_key" in video_service.include_params, (
                f"{video_service.value} should include api_key param"
            )


class TestExternalTurnDetection:
    """Test uses_external_turn_detection and its effect on generated imports."""

    def test_standard_stt_does_not_use_external_turn(self):
        assert ServiceLoader.uses_external_turn_detection("deepgram_stt") is False

    def test_external_turn_stt_services(self):
        # Services that drive their own end-of-turn detection.
        assert ServiceLoader.uses_external_turn_detection("deepgram_flux_stt") is True
        assert ServiceLoader.uses_external_turn_detection("cartesia_turns_stt") is True

    def test_none_and_unknown_stt(self):
        assert ServiceLoader.uses_external_turn_detection(None) is False
        assert ServiceLoader.uses_external_turn_detection("nonexistent_stt") is False

    def test_external_turn_strategies_import_branch(self):
        """ExternalUserTurnStrategies is imported only for external-turn STT services."""
        base = {"llm": "openai_llm", "tts": "cartesia_tts"}

        standard = "\n".join(
            ServiceLoader.get_imports_for_services(
                {"transports": ["daily"], "stt": "deepgram_stt", **base}, {}, "web"
            )
        )
        assert "ExternalUserTurnStrategies" not in standard

        external = "\n".join(
            ServiceLoader.get_imports_for_services(
                {"transports": ["daily"], "stt": "deepgram_flux_stt", **base}, {}, "web"
            )
        )
        assert "ExternalUserTurnStrategies" in external


class TestTransportImportBranching:
    """Test the dial-out / SIP branching in get_imports_for_services.

    Most transports build via create_transport() and queue an LLMRunFrame on
    connect. Dial-out and Twilio+Daily SIP keep a bespoke hand-built transport,
    and dial-out additionally waits for the callee instead of kicking off the LLM.
    """

    BASE = {"stt": "deepgram_stt", "llm": "openai_llm", "tts": "cartesia_tts"}

    def _imports(self, transport):
        services = {"transports": [transport], **self.BASE}
        return "\n".join(ServiceLoader.get_imports_for_services(services, {}, "web"))

    @pytest.mark.parametrize(
        "transport,expects_create_transport",
        [
            ("daily", True),
            ("daily_pstn_dialin", True),  # dial-in is collapsed onto create_transport
            ("daily_pstn_dialout", False),  # bespoke hand-built transport
            ("twilio_daily_sip_dialin", False),
            ("twilio_daily_sip_dialout", False),
        ],
    )
    def test_create_transport_import_branch(self, transport, expects_create_transport):
        imports = self._imports(transport)
        # create_transport is imported as a bare symbol; check the import line.
        has_import = any(
            line.strip().endswith("create_transport") or "import create_transport" in line
            for line in imports.splitlines()
        )
        assert has_import is expects_create_transport

    @pytest.mark.parametrize(
        "transport,expects_llm_run_frame",
        [
            ("daily", True),
            ("daily_pstn_dialin", True),
            ("twilio_daily_sip_dialin", True),  # dial-in still kicks off the LLM
            ("daily_pstn_dialout", False),  # dial-out waits for the callee
            ("twilio_daily_sip_dialout", False),
        ],
    )
    def test_llm_run_frame_import_branch(self, transport, expects_llm_run_frame):
        imports = self._imports(transport)
        assert ("LLMRunFrame" in imports) is expects_llm_run_frame
