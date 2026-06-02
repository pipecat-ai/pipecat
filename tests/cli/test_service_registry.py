"""Tests for service registry integrity and completeness."""

import pytest

from pipecat.cli.registry import ServiceLoader, ServiceRegistry


class TestServiceRegistryIntegrity:
    """Test that the service registry is complete and consistent."""

    def test_all_services_have_configs(self):
        """Verify all services have corresponding configs."""
        missing = ServiceLoader.get_missing_services()

        if missing["missing_configs"]:
            pytest.fail(
                f"Services missing configs: {', '.join(missing['missing_configs'])}\n"
                f"Add these services to src/pipecat/cli/registry/_configs.py"
            )

    def test_all_services_have_imports(self):
        """Verify all services have corresponding import statements."""
        missing = ServiceLoader.get_missing_services()

        if missing["missing_imports"]:
            pytest.fail(
                f"Services missing imports: {', '.join(missing['missing_imports'])}\n"
                f"Add these services to ServiceRegistry.IMPORTS in services.py"
            )

    def test_service_definitions_are_valid(self):
        """Verify all service definitions have required fields.

        This test validates that all services were created successfully
        (which means they passed __post_init__ validation).
        """
        all_services = []
        all_services.extend(ServiceRegistry.WEBRTC_TRANSPORTS)
        all_services.extend(ServiceRegistry.TELEPHONY_TRANSPORTS)
        all_services.extend(ServiceRegistry.STT_SERVICES)
        all_services.extend(ServiceRegistry.LLM_SERVICES)
        all_services.extend(ServiceRegistry.TTS_SERVICES)
        all_services.extend(ServiceRegistry.REALTIME_SERVICES)
        all_services.extend(ServiceRegistry.VIDEO_SERVICES)

        # Verify all services have required fields
        for service in all_services:
            assert service.value, f"Service missing value"
            assert service.label, f"Service {service.value} missing label"
            assert service.package, f"Service {service.value} missing package"

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
    def test_every_service_has_config(self, service):
        """Test that every single service has a valid config."""
        service_value = service.value
        config = ServiceLoader.get_service_config(service_value)

        assert config is not None, f"Service {service_value} missing config in _configs.py"
        assert len(config.strip()) > 0, f"Service {service_value} has empty config"
        # Config should contain the service class name (e.g., "DeepgramSTTService")
        assert "Service" in config or "LLM" in config, (
            f"Service {service_value} config doesn't look like valid service initialization code"
        )

    @pytest.mark.parametrize(
        "service",
        ServiceRegistry.STT_SERVICES
        + ServiceRegistry.LLM_SERVICES
        + ServiceRegistry.TTS_SERVICES
        + ServiceRegistry.REALTIME_SERVICES
        + ServiceRegistry.VIDEO_SERVICES,
        ids=lambda s: s.value,
    )
    def test_every_service_has_imports(self, service):
        """Test that every single service has import statements."""
        service_value = service.value
        imports = ServiceLoader.get_service_import(service_value)

        assert imports is not None, (
            f"Service {service_value} missing imports in ServiceRegistry.IMPORTS"
        )
        assert len(imports) > 0, f"Service {service_value} has empty imports list"
        # At least one import should reference pipecat
        assert any("pipecat" in imp for imp in imports), (
            f"Service {service_value} imports don't reference pipecat"
        )

    @pytest.mark.parametrize(
        "transport",
        ServiceRegistry.WEBRTC_TRANSPORTS + ServiceRegistry.TELEPHONY_TRANSPORTS,
        ids=lambda t: t.value,
    )
    def test_every_transport_has_imports(self, transport):
        """Test that every transport has import statements."""
        transport_value = transport.value
        imports = ServiceLoader.get_service_import(transport_value)

        assert imports is not None, (
            f"Transport {transport_value} missing imports in ServiceRegistry.IMPORTS"
        )
        assert len(imports) > 0, f"Transport {transport_value} has empty imports list"
        # Transport imports should reference pipecat
        assert any("pipecat" in imp for imp in imports), (
            f"Transport {transport_value} imports don't reference pipecat"
        )

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
        assert len(observability_imports) == 2
        assert any("WhiskerObserver" in imp for imp in observability_imports)
        assert any("TailObserver" in imp for imp in observability_imports)

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
        assert "TailObserver" in import_str
        assert "pipecat_whisker" in import_str
        assert "pipecat_tail" in import_str

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
