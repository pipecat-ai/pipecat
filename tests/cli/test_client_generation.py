"""Integration tests for client generation."""

import pytest

from pipecat.cli.generators.project import ProjectGenerator
from pipecat.cli.prompts.questions import ProjectConfig


class TestClientGeneration:
    """Test client generation with different configurations."""

    def test_vite_with_multiple_transports(self, tmp_path):
        """Test Vite client generation with multiple transports."""
        config = ProjectConfig(
            project_name="test-vite-multi",
            bot_type="web",
            transports=["daily", "smallwebrtc"],
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            generate_client=True,
            client_framework="react",
            client_server="vite",
        )

        generator = ProjectGenerator(config)
        project_path = generator.generate(output_dir=tmp_path)

        # Verify structure
        assert (project_path / "server" / "bot.py").exists()
        assert (project_path / "client" / "src" / "config.ts").exists()
        assert (project_path / "client" / "src" / "main.tsx").exists()
        assert (project_path / "client" / "package.json").exists()

        # Verify config.ts content
        config_content = (project_path / "client" / "src" / "config.ts").read_text()
        assert "'daily'" in config_content
        assert "'smallwebrtc'" in config_content
        assert "AVAILABLE_TRANSPORTS" in config_content
        assert "DEFAULT_TRANSPORT" in config_content
        assert "TRANSPORT_PROPS" in config_content

        # Verify static TypeScript has no Jinja2 syntax
        main_tsx = (project_path / "client" / "src" / "main.tsx").read_text()
        assert "{%" not in main_tsx
        assert "TRANSPORT_PROPS" in main_tsx

    def test_nextjs_with_single_transport(self, tmp_path):
        """Test Next.js client generation with single transport."""
        config = ProjectConfig(
            project_name="test-nextjs-single",
            bot_type="web",
            transports=["daily"],
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            generate_client=True,
            client_framework="react",
            client_server="nextjs",
        )

        generator = ProjectGenerator(config)
        project_path = generator.generate(output_dir=tmp_path)

        # Verify structure
        assert (project_path / "client" / "src" / "config.ts").exists()
        assert (project_path / "client" / "src" / "app" / "page.tsx").exists()

        # Verify config.ts has only selected transport in available list
        config_content = (project_path / "client" / "src" / "config.ts").read_text()
        assert "'daily'" in config_content
        assert "AVAILABLE_TRANSPORTS" in config_content

    def test_server_only_no_client(self, tmp_path):
        """Test server-only generation (no client)."""
        config = ProjectConfig(
            project_name="test-server-only",
            bot_type="web",
            transports=["daily"],
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            generate_client=False,
        )

        generator = ProjectGenerator(config)
        project_path = generator.generate(output_dir=tmp_path)

        # Verify server exists, client doesn't
        assert (project_path / "server" / "bot.py").exists()
        assert not (project_path / "client").exists()

    def test_package_json_dependencies(self, tmp_path):
        """Test that package.json only includes selected transport packages."""
        config = ProjectConfig(
            project_name="test-deps",
            bot_type="web",
            transports=["daily"],  # Only daily
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            generate_client=True,
            client_framework="react",
            client_server="vite",
        )

        generator = ProjectGenerator(config)
        project_path = generator.generate(output_dir=tmp_path)

        package_json = (project_path / "client" / "package.json").read_text()

        # Should include daily transport
        assert "@pipecat-ai/daily-transport" in package_json

        # Should NOT include other transports
        assert "@pipecat-ai/small-webrtc-transport" not in package_json

    def test_gitignore_includes_client(self, tmp_path):
        """Test that .gitignore includes client directories when client is generated."""
        config = ProjectConfig(
            project_name="test-gitignore",
            bot_type="web",
            transports=["daily"],
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            generate_client=True,
            client_framework="react",
            client_server="vite",
        )

        generator = ProjectGenerator(config)
        project_path = generator.generate(output_dir=tmp_path)

        gitignore = (project_path / ".gitignore").read_text()
        assert "client/node_modules" in gitignore
        assert "client/dist" in gitignore or "client/.next" in gitignore

    def test_readme_reflects_structure(self, tmp_path):
        """Test that README shows correct project structure."""
        config = ProjectConfig(
            project_name="test-readme",
            bot_type="web",
            transports=["daily"],
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            generate_client=True,
            client_framework="react",
            client_server="vite",
        )

        generator = ProjectGenerator(config)
        project_path = generator.generate(output_dir=tmp_path)

        readme = (project_path / "README.md").read_text()
        assert "server/" in readme
        assert "client/" in readme
        assert "npm install" in readme or "npm run dev" in readme


class TestStaticTypeScript:
    """Test that generated TypeScript is clean and static."""

    def test_no_jinja2_in_static_files(self, tmp_path):
        """Verify static TypeScript files don't contain Jinja2 syntax."""
        config = ProjectConfig(
            project_name="test-static",
            bot_type="web",
            transports=["daily", "smallwebrtc"],
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            generate_client=True,
            client_framework="react",
            client_server="vite",
        )

        generator = ProjectGenerator(config)
        project_path = generator.generate(output_dir=tmp_path)

        # Check static files
        static_files = [
            "client/src/main.tsx",
            "client/src/components/App.tsx",
            "client/src/components/TransportSelect.tsx",
        ]

        for file_path in static_files:
            content = (project_path / file_path).read_text()
            # Should not have Jinja2 template syntax
            assert "{%" not in content, f"{file_path} contains Jinja2 syntax"
            assert "endfor" not in content, f"{file_path} contains Jinja2 syntax"

    def test_all_transport_types_in_config(self, tmp_path):
        """Test that config.ts includes all transport types even if not selected."""
        config = ProjectConfig(
            project_name="test-all-transports",
            bot_type="web",
            transports=["daily"],  # Only select one
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            generate_client=True,
            client_framework="react",
            client_server="vite",
        )

        generator = ProjectGenerator(config)
        project_path = generator.generate(output_dir=tmp_path)

        config_content = (project_path / "client" / "src" / "config.ts").read_text()

        # Type definition and TRANSPORT_CONFIG should include ALL web transports the
        # registry knows about, regardless of which one was selected. Derive the
        # expected set from the registry so new web transports don't silently drift.
        from pipecat.cli.registry import ServiceRegistry

        web_transport_values = [t.value for t in ServiceRegistry.WEBRTC_TRANSPORTS]
        assert len(web_transport_values) >= 2  # sanity: more than just the selected one
        for value in web_transport_values:
            assert f"'{value}'" in config_content, f"{value} missing from TransportType union"
            assert f"{value}:" in config_content, f"{value} missing from TRANSPORT_CONFIG"

        # But AVAILABLE_TRANSPORTS should only have selected ones
        lines = config_content.split("\n")
        available_lines = []
        in_available = False
        for line in lines:
            if "AVAILABLE_TRANSPORTS" in line:
                in_available = True
            elif in_available:
                if "]" in line:
                    break
                available_lines.append(line)

        available_text = "\n".join(available_lines)
        assert "'daily'" in available_text
        assert "'smallwebrtc'" not in available_text  # Not selected


class TestMonorepoStructure:
    """Test the monorepo directory structure."""

    def test_monorepo_structure_with_client(self, tmp_path):
        """Test that projects with clients have correct monorepo structure."""
        config = ProjectConfig(
            project_name="test-monorepo",
            bot_type="web",
            transports=["daily"],
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            generate_client=True,
            client_framework="react",
            client_server="vite",
            deploy_to_cloud=True,
        )

        generator = ProjectGenerator(config)
        project_path = generator.generate(output_dir=tmp_path)

        # Root level files
        assert (project_path / "README.md").exists()
        assert (project_path / ".gitignore").exists()

        # Server files
        assert (project_path / "server" / "bot.py").exists()
        assert (project_path / "server" / "pyproject.toml").exists()
        assert (project_path / "server" / ".env.example").exists()
        assert (project_path / "server" / "Dockerfile").exists()
        assert (project_path / "server" / "pcc-deploy.toml").exists()

        # Client files
        assert (project_path / "client" / "package.json").exists()
        assert (project_path / "client" / "src" / "main.tsx").exists()
        assert (project_path / "client" / "src" / "config.ts").exists()

    def test_server_only_structure(self, tmp_path):
        """Test that server-only projects don't create client directory."""
        config = ProjectConfig(
            project_name="test-server-structure",
            bot_type="web",
            transports=["daily"],
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            generate_client=False,
        )

        generator = ProjectGenerator(config)
        project_path = generator.generate(output_dir=tmp_path)

        # Root level files
        assert (project_path / "README.md").exists()
        assert (project_path / ".gitignore").exists()

        # Server files
        assert (project_path / "server" / "bot.py").exists()
        assert (project_path / "server" / "pyproject.toml").exists()

        # NO client directory
        assert not (project_path / "client").exists()


class TestVanillaJSGeneration:
    """Test Vanilla JS client generation."""

    def test_vanilla_vite_generation(self, tmp_path):
        """Test Vanilla JS client generation (always uses Vite)."""
        config = ProjectConfig(
            project_name="test-vanilla-vite",
            bot_type="web",
            transports=["daily", "smallwebrtc"],
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            generate_client=True,
            client_framework="vanilla",
            client_server="vite",
        )

        generator = ProjectGenerator(config)
        project_path = generator.generate(output_dir=tmp_path)

        # Verify client structure
        assert (project_path / "client" / "src" / "config.js").exists()
        assert (project_path / "client" / "src" / "app.js").exists()
        assert (project_path / "client" / "index.html").exists()
        assert (project_path / "client" / "package.json").exists()

        # Verify config.js content
        config_content = (project_path / "client" / "src" / "config.js").read_text()
        assert "'daily'" in config_content
        assert "'smallwebrtc'" in config_content
        assert "AVAILABLE_TRANSPORTS" in config_content

        # Verify app.js has no Jinja2 syntax
        app_js = (project_path / "client" / "src" / "app.js").read_text()
        assert "{%" not in app_js
        assert "AVAILABLE_TRANSPORTS" in app_js
        assert "from './config'" in app_js

    def test_vanilla_package_json_dependencies(self, tmp_path):
        """Test that Vanilla JS package.json only includes selected transports."""
        config = ProjectConfig(
            project_name="test-vanilla-deps",
            bot_type="web",
            transports=["daily"],  # Only daily
            mode="cascade",
            stt_service="deepgram_stt",
            llm_service="openai_llm",
            tts_service="cartesia_tts",
            generate_client=True,
            client_framework="vanilla",
            client_server="vite",
        )

        generator = ProjectGenerator(config)
        project_path = generator.generate(output_dir=tmp_path)

        package_json = (project_path / "client" / "package.json").read_text()

        # Should include daily transport
        assert "@pipecat-ai/daily-transport" in package_json

        # Should NOT include other transports
        assert "@pipecat-ai/small-webrtc-transport" not in package_json


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
