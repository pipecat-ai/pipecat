"""Tests for interactive prompts and question flow."""

from questionary import Choice

from pipecat.cli.prompts.questions import ProjectConfig
from pipecat.cli.registry import ServiceLoader, ServiceRegistry


class TestProjectConfig:
    """Tests for ProjectConfig dataclass."""

    def test_project_config_creation(self):
        """Test that ProjectConfig can be created with required fields."""
        config = ProjectConfig(
            project_name="test-bot",
            bot_type="web",
            transports=["daily"],
            mode="cascade",
        )
        assert config.project_name == "test-bot"
        assert config.bot_type == "web"
        assert config.transports == ["daily"]
        assert config.mode == "cascade"

    def test_project_config_defaults(self):
        """Test that ProjectConfig has sensible defaults."""
        config = ProjectConfig(
            project_name="test-bot",
            bot_type="web",
        )
        assert config.transports == []
        assert config.mode == "cascade"
        assert config.stt_service is None
        assert config.llm_service is None
        assert config.tts_service is None
        assert config.realtime_service is None
        assert config.video_service is None
        assert config.video_input is False
        assert config.video_output is False
        assert config.recording is False
        assert config.transcription is False
        assert config.deploy_to_cloud is False
        assert config.enable_krisp is False
        assert config.enable_observability is False
        # Client-related defaults
        assert config.generate_client is False
        assert config.client_framework is None
        assert config.client_server is None

    def test_project_config_with_client(self):
        """Test that ProjectConfig can be created with client fields."""
        config = ProjectConfig(
            project_name="test-bot",
            bot_type="web",
            transports=["daily"],
            mode="cascade",
            generate_client=True,
            client_framework="react",
            client_server="vite",
        )
        assert config.generate_client is True
        assert config.client_framework == "react"
        assert config.client_server == "vite"

    def test_project_config_nextjs_client(self):
        """Test that ProjectConfig works with Next.js client."""
        config = ProjectConfig(
            project_name="test-nextjs-bot",
            bot_type="web",
            transports=["daily"],
            mode="cascade",
            generate_client=True,
            client_framework="react",
            client_server="nextjs",
        )
        assert config.generate_client is True
        assert config.client_framework == "react"
        assert config.client_server == "nextjs"

    def test_project_config_with_video_service(self):
        """Test that ProjectConfig can be created with a video service."""
        config = ProjectConfig(
            project_name="test-video-bot",
            bot_type="web",
            transports=["daily"],
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            video_service="tavus_video",
            video_output=True,
        )
        assert config.video_service == "tavus_video"
        assert config.video_output is True
        assert config.bot_type == "web"  # Video services only for web bots


class TestServiceDefinitionChoiceCreation:
    """
    Regression tests to ensure ServiceDefinition objects can be used to create
    questionary Choice objects. This would have caught the dataclass migration bug.
    """

    def test_service_definitions_work_as_choices(self):
        """Every service/transport ServiceDefinition can build a questionary Choice.

        Guards the original dataclass-migration bug across all registry lists at once:
        ``Choice(title=svc.label, value=svc.value)`` must not raise and must yield a
        usable Choice for every entry.
        """
        all_services = (
            ServiceRegistry.STT_SERVICES
            + ServiceRegistry.LLM_SERVICES
            + ServiceRegistry.TTS_SERVICES
            + ServiceRegistry.REALTIME_SERVICES
            + ServiceRegistry.VIDEO_SERVICES
            + ServiceRegistry.WEBRTC_TRANSPORTS
            + ServiceRegistry.TELEPHONY_TRANSPORTS
        )
        assert len(all_services) > 0
        for svc in all_services:
            choice = Choice(title=svc.label, value=svc.value)
            assert choice.title == svc.label
            assert choice.value == svc.value

    def test_transport_options_query(self):
        """get_transport_options() returns the right Choices for each bot type."""
        web = ServiceLoader.get_transport_options("web")
        assert any("Daily" in svc.label for svc in web)
        telephony = ServiceLoader.get_transport_options("telephony")
        assert any("Twilio" in svc.label for svc in telephony)

    def test_transport_fallback_access(self):
        """Test that transport options can be accessed by index (for fallback logic)."""
        transport_options = ServiceLoader.get_transport_options("web")
        assert len(transport_options) > 0
        # This is used in the fallback logic: transport_options[0].value
        first_transport = transport_options[0]
        assert hasattr(first_transport, "value")
        assert hasattr(first_transport, "label")
        assert isinstance(first_transport.value, str)
        assert isinstance(first_transport.label, str)

    def test_all_services_have_required_fields(self):
        """Test that all services have the fields needed for Choice creation."""
        all_services = (
            ServiceRegistry.STT_SERVICES
            + ServiceRegistry.LLM_SERVICES
            + ServiceRegistry.TTS_SERVICES
            + ServiceRegistry.REALTIME_SERVICES
            + ServiceRegistry.VIDEO_SERVICES
            + ServiceRegistry.WEBRTC_TRANSPORTS
            + ServiceRegistry.TELEPHONY_TRANSPORTS
        )

        for service in all_services:
            # These are the fields used in questions.py
            assert hasattr(service, "label"), f"Service missing 'label': {service}"
            assert hasattr(service, "value"), f"Service missing 'value': {service}"
            assert isinstance(service.label, str), f"Service label not a string: {service}"
            assert isinstance(service.value, str), f"Service value not a string: {service}"
            assert len(service.label) > 0, f"Service has empty label: {service}"
            assert len(service.value) > 0, f"Service has empty value: {service}"
