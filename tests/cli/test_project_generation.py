"""Integration tests for project generation with different configurations."""

import ast
import shutil
import subprocess

import pytest

from pipecat.cli.generators.project import ProjectGenerator
from pipecat.cli.prompts.questions import ProjectConfig


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test projects."""
    output_dir = tmp_path / "test_projects"
    output_dir.mkdir()
    yield output_dir
    # Cleanup happens automatically with tmp_path

    # For debugging: Uncomment these lines to preserve test files
    # from pathlib import Path
    # output_dir = Path("/tmp/pipecat-cli-test-projects")
    # output_dir.mkdir(parents=True, exist_ok=True)
    # print(f"\n✅ Test files preserved at: {output_dir}")
    # return  # Skip cleanup


def validate_python_syntax(file_path):
    """Validate that a Python file has valid syntax."""
    with open(file_path) as f:
        code = f.read()
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def validate_pyproject_toml(file_path):
    """Validate that pyproject.toml is valid TOML and has required fields."""
    import tomllib

    with open(file_path, "rb") as f:
        data = tomllib.load(f)

    # Check required fields
    assert "project" in data, "pyproject.toml must have [project] section"
    assert "name" in data["project"], "project.name is required"
    assert "dependencies" in data["project"], "project.dependencies is required"
    assert len(data["project"]["dependencies"]) > 0, "Must have at least one dependency"

    # Check for pipecat-ai dependency
    has_pipecat = any("pipecat-ai" in dep for dep in data["project"]["dependencies"])
    assert has_pipecat, "Must include pipecat-ai dependency"

    return True


def validate_imports_resolvable(bot_file_path):
    """Check that all imports in bot.py could theoretically resolve."""
    with open(bot_file_path) as f:
        tree = ast.parse(f.read())

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    # Check that we have the expected Pipecat imports
    has_pipecat = any("pipecat" in imp for imp in imports)
    assert has_pipecat, "bot.py should import from pipecat"

    return True, imports


def assert_server_ruff_clean(server_path):
    """Assert generated Python under server_path is already Ruff-formatted.

    Uses the Ruff binary bundled with the CLI's dependencies (the same one
    ``ProjectGenerator._format_python_files`` resolves), so the check does not
    depend on ``ruff`` being on PATH.
    """
    from ruff.__main__ import find_ruff_bin

    result = subprocess.run(
        [find_ruff_bin(), "format", "--check", str(server_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Generated Python under {server_path} is not Ruff-formatted:\n"
        f"{result.stdout}{result.stderr}"
    )


def assert_server_ruff_lint_clean(server_path):
    """Assert generated Python under server_path is free of Pyflakes lint errors.

    Runs ``ruff check --select F`` (the same bundled binary used for formatting):
    no unused imports, no f-strings without placeholders, no undefined/redefined
    names, etc. The generator only runs ``ruff check --select I`` (import sorting),
    so this is the guard that the templates produce genuinely clean code.
    """
    from ruff.__main__ import find_ruff_bin

    result = subprocess.run(
        [find_ruff_bin(), "check", "--select", "F", str(server_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Generated Python under {server_path} has lint errors:\n{result.stdout}{result.stderr}"
    )


# Test configurations for different transport types
TEST_CONFIGS = [
    # WebRTC Transports - Cascade
    {
        "name": "daily-cascade",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
    },
    {
        "name": "smallwebrtc-cascade",
        "bot_type": "web",
        "transports": ["smallwebrtc"],
        "mode": "cascade",
        "stt_service": "assemblyai_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
    },
    # Telephony Transports - Cascade
    {
        "name": "twilio-cascade",
        "bot_type": "telephony",
        "transports": ["twilio"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
    },
    {
        "name": "telnyx-cascade",
        "bot_type": "telephony",
        "transports": ["telnyx"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
    },
    # Realtime Pipelines
    {
        "name": "daily-realtime",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "realtime",
        "realtime_service": "openai_realtime",
    },
    {
        "name": "smallwebrtc-realtime",
        "bot_type": "web",
        "transports": ["smallwebrtc"],
        "mode": "realtime",
        "realtime_service": "gemini_live_realtime",
    },
    # Mixed Transports (Telephony + WebRTC)
    {
        "name": "twilio-webrtc-mixed",
        "bot_type": "telephony",
        "transports": ["twilio", "smallwebrtc"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
    },
    # With features
    {
        "name": "daily-with-features",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
        "video_input": True,
        "video_output": True,
        "recording": True,
        "transcription": True,
        "deploy_to_cloud": True,
        "enable_krisp": True,
    },
    # With observability
    {
        "name": "daily-with-observability",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
        "enable_observability": True,
    },
    # More telephony providers
    {
        "name": "plivo-cascade",
        "bot_type": "telephony",
        "transports": ["plivo"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "anthropic_llm",
        "tts_service": "elevenlabs_tts",
    },
    {
        "name": "exotel-cascade",
        "bot_type": "telephony",
        "transports": ["exotel"],
        "mode": "cascade",
        "stt_service": "assemblyai_stt",
        "llm_service": "groq_llm",
        "tts_service": "rime_tts",
    },
    # More realtime services
    {
        "name": "azure-realtime",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "realtime",
        "realtime_service": "azure_realtime",
    },
    {
        "name": "aws-nova-realtime",
        "bot_type": "web",
        "transports": ["smallwebrtc"],
        "mode": "realtime",
        "realtime_service": "aws_nova_realtime",
    },
    # Different service combinations for cascade
    {
        "name": "google-vertex-stack",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "cascade",
        "stt_service": "google_stt",
        "llm_service": "google_vertex_llm",
        "tts_service": "google_tts",
    },
    {
        "name": "azure-stack",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "cascade",
        "stt_service": "azure_stt",
        "llm_service": "azure_llm",
        "tts_service": "azure_tts",
    },
    {
        "name": "aws-stack",
        "bot_type": "web",
        "transports": ["smallwebrtc"],
        "mode": "cascade",
        "stt_service": "aws_transcribe_stt",
        "llm_service": "aws_bedrock_llm",
        "tts_service": "aws_polly_tts",
    },
    # Recording and transcription features
    {
        "name": "daily-recording",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
        "recording": True,
    },
    {
        "name": "daily-transcription",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "elevenlabs_tts",
        "transcription": True,
    },
    # Cloud deployment variations
    {
        "name": "twilio-cloud-no-krisp",
        "bot_type": "telephony",
        "transports": ["twilio"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
        "deploy_to_cloud": True,
        "enable_krisp": False,
    },
    {
        "name": "smallwebrtc-cloud-with-video",
        "bot_type": "web",
        "transports": ["smallwebrtc"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
        "video_input": True,
        "video_output": True,
        "deploy_to_cloud": True,
    },
    # Video combinations
    {
        "name": "daily-video-input-only",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "anthropic_llm",
        "tts_service": "cartesia_tts",
        "video_input": True,
        "video_output": False,
    },
    {
        "name": "smallwebrtc-video-output-only",
        "bot_type": "web",
        "transports": ["smallwebrtc"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "elevenlabs_tts",
        "video_input": False,
        "video_output": True,
    },
    # Multiple transports with different features
    {
        "name": "daily-smallwebrtc-multi",
        "bot_type": "web",
        "transports": ["daily", "smallwebrtc"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
    },
    {
        "name": "telnyx-daily-mixed-cloud",
        "bot_type": "telephony",
        "transports": ["telnyx", "daily"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
        "deploy_to_cloud": True,
        "enable_krisp": True,
    },
    # Video avatar services
    {
        "name": "daily-tavus-video",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
        "video_service": "tavus_video",
        "video_output": True,
    },
    {
        "name": "smallwebrtc-heygen-video",
        "bot_type": "web",
        "transports": ["smallwebrtc"],
        "mode": "cascade",
        "stt_service": "assemblyai_stt",
        "llm_service": "anthropic_llm",
        "tts_service": "elevenlabs_tts",
        "video_service": "heygen_video",
        "video_output": True,
    },
    {
        "name": "daily-simli-video",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
        "video_service": "simli_video",
        "video_output": True,
    },
    {
        "name": "daily-tavus-video-cloud",
        "bot_type": "web",
        "transports": ["daily"],
        "mode": "cascade",
        "stt_service": "deepgram_stt",
        "llm_service": "openai_llm",
        "tts_service": "cartesia_tts",
        "video_service": "tavus_video",
        "video_output": True,
        "deploy_to_cloud": True,
        "enable_krisp": True,
    },
]


@pytest.mark.parametrize("config_data", TEST_CONFIGS, ids=lambda c: c["name"])
def test_project_generation(config_data, temp_output_dir):
    """Test that projects generate successfully with different configurations."""
    # Create config with defaults for optional fields
    config = ProjectConfig(
        project_name=config_data["name"],
        bot_type=config_data["bot_type"],
        transports=config_data["transports"],
        mode=config_data["mode"],
        stt_service=config_data.get("stt_service"),
        llm_service=config_data.get("llm_service"),
        tts_service=config_data.get("tts_service"),
        realtime_service=config_data.get("realtime_service"),
        video_service=config_data.get("video_service"),
        video_input=config_data.get("video_input", False),
        video_output=config_data.get("video_output", False),
        recording=config_data.get("recording", False),
        transcription=config_data.get("transcription", False),
        deploy_to_cloud=config_data.get("deploy_to_cloud", False),
        enable_krisp=config_data.get("enable_krisp", False),
        enable_observability=config_data.get("enable_observability", False),
    )

    # Generate project
    generator = ProjectGenerator(config)

    # Ensure clean output directory
    project_path = temp_output_dir / config.project_name
    if project_path.exists():
        shutil.rmtree(project_path)

    # Generate into temp directory
    generator.generate(output_dir=temp_output_dir)

    # Verify core files exist (in monorepo structure)
    assert (project_path / "server" / "bot.py").exists(), "server/bot.py should exist"
    assert (project_path / "server" / "pyproject.toml").exists(), (
        "server/pyproject.toml should exist"
    )
    assert (project_path / "server" / ".env.example").exists(), "server/.env.example should exist"
    assert (project_path / ".gitignore").exists(), ".gitignore should exist"
    assert (project_path / "README.md").exists(), "README.md should exist"

    # Verify cloud deployment files
    if config.deploy_to_cloud:
        assert (project_path / "server" / "Dockerfile").exists(), (
            "server/Dockerfile should exist for cloud deployment"
        )
        assert (project_path / "server" / "pcc-deploy.toml").exists(), (
            "server/pcc-deploy.toml should exist for cloud deployment"
        )

    # Validate Python syntax
    bot_file = project_path / "server" / "bot.py"
    is_valid, error = validate_python_syntax(bot_file)
    assert is_valid, f"bot.py has syntax errors: {error}"

    # Validate imports are resolvable
    is_valid, imports = validate_imports_resolvable(bot_file)
    assert is_valid, "bot.py should have valid Pipecat imports"

    # Generated Python must be Ruff-formatted. Regression guard: the formatting
    # step used to silently no-op when `ruff` was not on PATH, shipping the raw
    # (mis-indented, unsorted-imports) template output.
    assert_server_ruff_clean(project_path / "server")

    # Generated Python must also be Pyflakes-clean (no unused imports, no
    # placeholder-less f-strings, etc.) — the generator only sorts imports.
    assert_server_ruff_lint_clean(project_path / "server")

    # Verify bot.py structure
    bot_content = bot_file.read_text()
    assert "async def run_bot" in bot_content, "bot.py should have run_bot function"
    assert "async def bot" in bot_content, "bot.py should have bot function"

    # Verify the selected services' actual classes appear in the generated bot,
    # rather than a weak substring like "stt". The expected class names come from
    # the registry, so this stays correct as services are added or renamed.
    from pipecat.cli.registry import ServiceLoader, ServiceRegistry

    def assert_service_class_referenced(value, service_lists):
        for service_list in service_lists:
            svc = ServiceLoader.get_service_by_value(service_list, value)
            if svc:
                primary_class = svc.class_name[0]
                assert primary_class in bot_content, (
                    f"{value} should reference {primary_class} in bot.py"
                )
                return
        pytest.fail(f"Service {value} not found in registry")

    if config.mode == "cascade":
        if config.stt_service:
            assert_service_class_referenced(config.stt_service, [ServiceRegistry.STT_SERVICES])
        if config.llm_service:
            assert_service_class_referenced(config.llm_service, [ServiceRegistry.LLM_SERVICES])
        if config.tts_service:
            assert_service_class_referenced(config.tts_service, [ServiceRegistry.TTS_SERVICES])
    elif config.mode == "realtime":
        if config.realtime_service:
            assert_service_class_referenced(
                config.realtime_service, [ServiceRegistry.REALTIME_SERVICES]
            )

    # Verify transport-specific code
    for transport in config.transports:
        if transport == "daily":
            assert "DailyParams" in bot_content or "daily" in bot_content.lower()
        elif transport == "smallwebrtc":
            assert "TransportParams" in bot_content or "webrtc" in bot_content.lower()
        elif transport in ["twilio", "telnyx", "plivo", "exotel"]:
            assert "FastAPIWebsocketParams" in bot_content, (
                f"{transport} should use FastAPIWebsocketParams"
            )

    # Validate pyproject.toml syntax and structure
    pyproject_file = project_path / "server" / "pyproject.toml"
    validate_pyproject_toml(pyproject_file)

    # Verify pyproject.toml has correct dependencies
    pyproject_content = pyproject_file.read_text()
    assert "pipecat-ai[" in pyproject_content, "pyproject.toml should have pipecat-ai dependencies"

    # Verify transport extras
    for transport in config.transports:
        if transport == "daily":
            assert "daily" in pyproject_content
        elif transport == "smallwebrtc":
            assert "webrtc" in pyproject_content

    # Verify cloud deployment extras
    if config.deploy_to_cloud:
        assert "pipecatcloud" in pyproject_content

    # Verify observability dependencies
    if config.enable_observability:
        assert "pipecat-ai-whisker" in pyproject_content
        assert "pipecat-ai-tail" in pyproject_content
        # Verify observability imports in bot.py
        assert "WhiskerObserver" in bot_content
        assert "TailObserver" in bot_content
        assert "from pipecat_whisker import WhiskerObserver" in bot_content
        assert "from pipecat_tail.observer import TailObserver" in bot_content

    # Verify video service dependencies and imports
    if config.video_service:
        # Video services should be in bot.py
        if config.video_service == "tavus_video":
            assert "TavusVideoService" in bot_content, "TavusVideoService should be imported"
            assert "tavus" in pyproject_content, "tavus extra should be in dependencies"
        elif config.video_service == "heygen_video":
            assert "HeyGenVideoService" in bot_content, "HeyGenVideoService should be imported"
            assert "heygen" in pyproject_content, "heygen extra should be in dependencies"
            assert "LiveAvatarNewSessionRequest" in bot_content, (
                "LiveAvatarNewSessionRequest should be imported for HeyGen"
            )
            assert "ServiceType" in bot_content, "ServiceType should be imported for HeyGen"
        elif config.video_service == "simli_video":
            assert "SimliVideoService" in bot_content, "SimliVideoService should be imported"
            assert "simli" in pyproject_content, "simli extra should be in dependencies"

        # Video service should be initialized in bot.py
        assert "video" in bot_content.lower(), "video service variable should be present"


def test_project_name_conflict(temp_output_dir):
    """Test that project generation handles name conflicts."""
    config = ProjectConfig(
        project_name="test-project",
        bot_type="web",
        transports=["daily"],
        mode="cascade",
        stt_service="deepgram_stt",
        llm_service="openai_llm",
        tts_service="cartesia_tts",
    )

    generator = ProjectGenerator(config)
    project_path = temp_output_dir / config.project_name

    # Generate first project
    generator.generate(output_dir=temp_output_dir)
    assert project_path.exists()
    assert (project_path / "server" / "bot.py").exists()

    # Note: The CLI prompts for a new name on conflict, but we can't test
    # that interactively here. This test just verifies the first generation works.


def test_generation_uses_utf8_on_windows_locale(monkeypatch, temp_output_dir):
    """Regression test for pipecat-ai/pipecat#4523.

    On Windows, ``Path.write_text(data)`` without an explicit ``encoding``
    falls back to the locale codec (cp1252), which cannot encode the ``→``
    characters in cascade-mode templates such as ``bot_cascade.py.jinja2``
    and ``README.md.jinja2``. This test simulates that environment by
    patching ``Path.write_text`` / ``Path.read_text`` to fail when
    ``encoding`` is omitted, then runs the quickstart configuration end to
    end and verifies the arrow-bearing files were written intact.
    """
    from pathlib import Path

    original_write = Path.write_text
    original_read = Path.read_text

    def patched_write(self, data, *args, **kwargs):
        if kwargs.get("encoding") is None and (len(args) == 0 or args[0] is None):
            # Simulate Windows cp1252 fallback — fails on '→' (U+2192).
            data.encode("cp1252")
        return original_write(self, data, *args, **kwargs)

    def patched_read(self, *args, **kwargs):
        if kwargs.get("encoding") is None and (len(args) == 0 or args[0] is None):
            # Reading a UTF-8 template containing '→' as cp1252 raises
            # UnicodeDecodeError on Windows. Force the same failure mode.
            return original_read(self, *args, encoding="cp1252", **kwargs)
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", patched_write)
    monkeypatch.setattr(Path, "read_text", patched_read)

    # Mirror the quickstart_command config from commands/init.py.
    config = ProjectConfig(
        project_name="pipecat-quickstart",
        bot_type="web",
        transports=["smallwebrtc", "daily"],
        mode="cascade",
        stt_service="deepgram_stt",
        llm_service="openai_responses_llm",
        tts_service="cartesia_tts",
        deploy_to_cloud=True,
    )
    ProjectGenerator(config).generate(output_dir=temp_output_dir, non_interactive=True)

    project = temp_output_dir / "pipecat-quickstart"
    bot = project / "server" / "bot.py"
    readme = project / "README.md"

    assert bot.exists(), "bot.py was not written"
    assert readme.exists(), "README.md was not written"
    # The cascade-mode template carries '→' in a code comment; confirm it
    # survived the cp1252-simulating environment.
    assert "→" in bot.read_text(encoding="utf-8")
    assert "→" in readme.read_text(encoding="utf-8")


def test_generated_python_formatted_without_ruff_on_path(monkeypatch, tmp_path, temp_output_dir):
    """Regression test: generated Python is formatted even when `ruff` is off PATH.

    When the CLI is installed as an isolated tool (``uv tool install`` /
    ``pipx``), only the ``pipecat``/``pc`` scripts land on PATH — Ruff's console
    script does not. ``_format_python_files`` must still run by resolving the
    bundled Ruff binary directly. Previously it shelled out to a bare ``ruff``
    and silently skipped formatting via ``FileNotFoundError``, shipping raw
    template output.
    """
    import shutil as _shutil

    # Point PATH at an empty directory so a bare `ruff` lookup fails on any
    # platform — mirroring an isolated tool install where ruff isn't exposed.
    empty_dir = tmp_path / "empty_path"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))
    assert _shutil.which("ruff") is None, "test setup: ruff should not be on PATH"

    config = ProjectConfig(
        project_name="fmt-no-path",
        bot_type="web",
        transports=["daily"],
        mode="cascade",
        stt_service="deepgram_stt",
        llm_service="openai_llm",
        tts_service="cartesia_tts",
    )
    ProjectGenerator(config).generate(output_dir=temp_output_dir, non_interactive=True)

    assert_server_ruff_clean(temp_output_dir / "fmt-no-path" / "server")


def test_format_warns_when_ruff_cannot_run(monkeypatch, temp_output_dir, capsys):
    """Regression test: a formatting failure warns instead of being silently swallowed.

    The old code caught ``FileNotFoundError`` and ``pass``ed, so a missing or
    broken Ruff produced unformatted output with no signal to the user.
    """
    config = ProjectConfig(
        project_name="warn-test",
        bot_type="web",
        transports=["daily"],
        mode="cascade",
        stt_service="deepgram_stt",
        llm_service="openai_llm",
        tts_service="cartesia_tts",
    )
    generator = ProjectGenerator(config)
    # Force Ruff resolution to a non-existent binary so the subprocess raises
    # FileNotFoundError — the case that used to be silently ignored.
    monkeypatch.setattr(generator, "_ruff_command", lambda: ["pipecat-cli-no-such-ruff-binary"])
    generator.generate(output_dir=temp_output_dir, non_interactive=True)

    captured = capsys.readouterr()
    assert "Could not run Ruff" in captured.out, (
        f"expected a Ruff warning on stdout, got:\n{captured.out}"
    )


def test_collapsed_transport_construction(temp_output_dir):
    """Standard cascade bots build transports via create_transport (no match/case)."""
    path = temp_output_dir / "collapsed"
    if path.exists():
        shutil.rmtree(path)
    ProjectGenerator(
        ProjectConfig(
            project_name="collapsed",
            bot_type="web",
            transports=["smallwebrtc"],
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
        )
    ).generate(output_dir=temp_output_dir)
    bot = (path / "server" / "bot.py").read_text()

    # Unified path: a transport_params dict + create_transport, no per-transport match/case.
    assert "transport_params = {" in bot
    assert "await create_transport(runner_args, transport_params)" in bot
    assert "match runner_args" not in bot
    assert "parse_telephony_websocket" not in bot

    # create_transport is imported.
    assert "from pipecat.runner.utils import create_transport" in bot

    ast.parse(bot)  # raises if the generated bot has a syntax error


def _gen_bot(temp_output_dir, name, **kwargs):
    """Generate a telephony cascade bot and return its bot.py text.

    Also asserts the generated server is formatting- and Pyflakes-clean, so the
    dial-in/dial-out/SIP bots (not covered by TEST_CONFIGS) are lint-guarded too.
    """
    path = temp_output_dir / name
    if path.exists():
        shutil.rmtree(path)
    ProjectGenerator(
        ProjectConfig(
            project_name=name,
            bot_type="telephony",
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            **kwargs,
        )
    ).generate(output_dir=temp_output_dir)
    assert_server_ruff_clean(path / "server")
    assert_server_ruff_lint_clean(path / "server")
    return (path / "server" / "bot.py").read_text()


def test_daily_pstn_dialin_uses_create_transport(temp_output_dir):
    """Daily PSTN dial-in is collapsed onto the unified create_transport path.

    Dial-in arrives as a typed DailyRunnerArguments and create_transport applies the
    dial-in settings from the request body, so the bot should NOT build a DailyTransport
    by hand. It still parses DailyDialinRequest for the optional personalization block."""
    bot = _gen_bot(
        temp_output_dir, "din", transports=["daily_pstn_dialin"], daily_pstn_mode="dial-in"
    )

    # Collapsed path
    assert "transport_params = {" in bot
    assert '"daily": lambda: DailyParams(' in bot
    assert "await create_transport(runner_args, transport_params)" in bot
    assert "await run_bot(transport, runner_args)" in bot

    # create_transport builds the transport — no hand-built DailyTransport / settings.
    assert "DailyTransport(" not in bot
    assert "DailyDialinSettings" not in bot

    # Active, guarded personalization using the typed DailyDialinRequest.
    assert "from pipecat.runner.types import DailyDialinRequest" in bot
    assert 'isinstance(runner_args.body, dict) and "dialin_settings" in runner_args.body' in bot
    assert "DailyDialinRequest.model_validate(runner_args.body)" in bot
    assert "request.dialin_settings.From" in bot

    ast.parse(bot)


def test_twilio_active_personalization_uses_call_info(temp_output_dir):
    """Twilio bots ship active personalization matching the examples: a typed CallInfo
    + get_call_info, read via attribute access (not commented, not dict .get)."""
    bot = _gen_bot(temp_output_dir, "tw", transports=["twilio"])

    # Typed helper (CallInfo model, not a dict)
    assert "class CallInfo(BaseModel):" in bot
    assert "async def get_call_info(call_sid: str | None) -> CallInfo | None:" in bot
    assert "from pydantic import BaseModel" in bot

    # Active (uncommented) personalization with attribute access
    assert "call_data = runner_args.call_data" in bot
    assert "call_info = await get_call_info(call_data.call_id) if call_data else None" in bot
    assert "call_info.from_number" in bot
    assert "call_info.get(" not in bot  # no dict-style access

    ast.parse(bot)


def test_dialout_and_sip_keep_bespoke_but_standard_run_bot(temp_output_dir):
    """Dial-out and SIP stay bespoke (room/token from the body, transport built by
    hand) but call the SAME standardized run_bot(transport, runner_args) — and the
    old per-flow run_bot signatures are gone."""
    scenarios = {
        "dout": dict(transports=["daily_pstn_dialout"], daily_pstn_mode="dial-out"),
        "sin": dict(transports=["twilio_daily_sip_dialin"], twilio_daily_sip_mode="dial-in"),
        "sout": dict(transports=["twilio_daily_sip_dialout"], twilio_daily_sip_mode="dial-out"),
    }
    for name, kwargs in scenarios.items():
        bot = _gen_bot(temp_output_dir, name, **kwargs)

        # One standardized signature + call site everywhere
        assert "async def run_bot(transport: BaseTransport, runner_args: RunnerArguments)" in bot
        assert "await run_bot(transport, runner_args)" in bot
        # Old run_bot signatures / call sites are gone (DialoutManager still takes
        # dialout_settings — that's the helper, not run_bot).
        assert "run_bot(\n        transport: BaseTransport, dialout_settings" not in bot
        assert "run_bot(transport, request.dialout_settings)" not in bot
        assert "run_bot(transport, request)" not in bot

        # Still bespoke: transport built by hand from the request body
        assert "DailyTransport(" in bot
        assert "AgentRequest.model_validate(runner_args.body)" in bot

        ast.parse(bot)


def test_run_bot_signature_uniform_across_modes(temp_output_dir):
    """Every generated bot — web, telephony, dial-out — shares one run_bot signature."""
    path = temp_output_dir / "webrb"
    if path.exists():
        shutil.rmtree(path)
    ProjectGenerator(
        ProjectConfig(
            project_name="webrb",
            bot_type="web",
            transports=["smallwebrtc"],
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
        )
    ).generate(output_dir=temp_output_dir)
    web = (path / "server" / "bot.py").read_text()
    dout = _gen_bot(
        temp_output_dir, "doutrb", transports=["daily_pstn_dialout"], daily_pstn_mode="dial-out"
    )
    sig = "async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:"
    assert sig in web
    assert sig in dout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
