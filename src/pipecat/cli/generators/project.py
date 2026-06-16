#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Main project generator that orchestrates file creation."""

import shutil
import subprocess
import sys
from pathlib import Path

import questionary
from jinja2 import Environment, PackageLoader, select_autoescape
from rich.console import Console

from pipecat.cli.prompts import ProjectConfig
from pipecat.cli.registry import ServiceLoader, ServiceRegistry

console = Console()


class ProjectGenerator:
    """Generates a complete Pipecat project from configuration."""

    def __init__(self, config: ProjectConfig):
        """Initialize the project generator.

        Args:
            config: Project configuration from user prompts
        """
        self.config = config
        self.env = Environment(
            loader=PackageLoader("pipecat.cli", "templates"),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _prompt_for_new_name(self, output_dir: Path) -> str:
        """Prompt user for a new project name if the current one already exists.

        Args:
            output_dir: The output directory where projects are created

        Returns:
            A valid project name that doesn't conflict with existing directories
        """
        while True:
            console.print(
                f"\n[yellow]⚠️  Directory '{self.config.project_name}' already exists![/yellow]"
            )
            new_name = questionary.text(
                "Please enter a different project name:",
                default=f"{self.config.project_name}-new",
                validate=lambda text: len(text) > 0 or "Project name cannot be empty",
            ).ask()

            if not new_name:
                raise KeyboardInterrupt("Project creation cancelled")

            # Check if the new name is available
            new_path = output_dir / new_name
            if not new_path.exists():
                return new_name

            # If still exists, loop will continue and prompt again

    def generate(
        self,
        output_dir: Path | None = None,
        non_interactive: bool = False,
        in_place: bool = False,
    ) -> Path:
        """Generate the complete project structure.

        Args:
            output_dir: Optional directory to create project in (defaults to current dir).
                When ``in_place`` is True this is the exact destination; otherwise the
                project is created in a ``<output_dir>/<project_name>`` subfolder.
            non_interactive: If True, raise FileExistsError instead of prompting
            in_place: If True, scaffold directly into ``output_dir`` (or the current
                directory) without nesting under a ``<project_name>`` subfolder.

        Returns:
            Path to the created project directory

        Raises:
            FileExistsError: If a project already exists at the destination and
                non_interactive is True
        """
        # Determine output directory
        if output_dir is None:
            output_dir = Path.cwd()

        if in_place:
            # Scaffold directly into output_dir — no <project_name> subfolder.
            project_path = output_dir
            project_path.mkdir(parents=True, exist_ok=True)
            # Guard against clobbering an existing project. The presence of a server/
            # directory is our "a project already lives here" signal (cargo-init style).
            if (project_path / "server").exists():
                if non_interactive:
                    raise FileExistsError(
                        f"A project already exists in {project_path} (found a 'server' directory)"
                    )
                console.print(
                    f"\n[yellow]⚠️  A project already exists in {project_path} "
                    f"(found a 'server' directory).[/yellow]"
                )
                raise KeyboardInterrupt("Project creation cancelled")
            return self._write_project_files(project_path)

        project_path = output_dir / self.config.project_name

        # Check if project already exists
        if project_path.exists():
            if non_interactive:
                raise FileExistsError(
                    f"Directory '{self.config.project_name}' already exists in {output_dir}"
                )
            new_name = self._prompt_for_new_name(output_dir)
            self.config.project_name = new_name
            project_path = output_dir / new_name

        return self._write_project_files(project_path)

    def _write_project_files(self, project_path: Path) -> Path:
        """Create the directory structure and render all project files.

        Args:
            project_path: The directory the project contents are written into.

        Returns:
            The same ``project_path``, after all files have been generated.
        """
        # Create project directory structure
        project_path.mkdir(parents=True, exist_ok=True)
        server_path = project_path / "server"
        server_path.mkdir(exist_ok=True)

        # Create client directory if generating client
        if self.config.generate_client:
            client_path = project_path / "client"
            client_path.mkdir(exist_ok=True)

        # Generate all files silently (operations are fast)
        # 1. Generate bot.py (in server/)
        self._generate_bot_file(server_path)

        # 1b. Generate server.py and server_utils.py for Daily PSTN and Twilio + Daily SIP
        if (
            "daily_pstn_dialin" in self.config.transports
            or "daily_pstn_dialout" in self.config.transports
            or "twilio_daily_sip_dialin" in self.config.transports
            or "twilio_daily_sip_dialout" in self.config.transports
        ):
            self._generate_server_files(server_path)

        # 2. Generate pyproject.toml (in server/)
        self._generate_pyproject(server_path)

        # 3. Generate .env.example (in server/)
        self._generate_env_example(server_path)

        # 3b. Generate starter eval scenarios (in server/evals/) if evals enabled
        if self.config.enable_eval:
            self._generate_eval_scenarios(server_path)

        # 4. Generate .gitignore (at root)
        self._generate_gitignore(project_path)

        # 5. Generate README.md (at root)
        self._generate_readme(project_path)

        # 6. Generate Dockerfile (in server/ if deploying to cloud)
        if self.config.deploy_to_cloud:
            self._generate_dockerfile(server_path)

        # 7. Generate pcc-deploy.toml (in server/ if deploying to cloud)
        if self.config.deploy_to_cloud:
            self._generate_pcc_deploy(server_path)

        # 8. Generate client (if requested)
        if self.config.generate_client:
            self._generate_client(project_path / "client")

        # Format generated Python files with Ruff
        self._format_python_files(server_path)

        return project_path

    def _generate_server_files(self, project_path: Path) -> None:
        """Generate server.py and server_utils.py for Daily PSTN dial-out or Twilio + Daily SIP."""
        # Determine which templates to use based on transport type and mode
        if self.config.daily_pstn_mode:
            # Daily PSTN - only dial-out has server files (dial-in doesn't need them)
            if self.config.daily_pstn_mode == "dial-in":
                # Daily PSTN dial-in doesn't require server.py/server_utils.py
                return
            mode = self.config.daily_pstn_mode  # 'dial-out'
            server_template_name = f"server/server_pstn_{mode.replace('-', '')}.py.jinja2"
            utils_template_name = f"server/server_utils_pstn_{mode.replace('-', '')}.py.jinja2"
        elif self.config.twilio_daily_sip_mode:
            # Twilio + Daily SIP
            mode = self.config.twilio_daily_sip_mode  # 'dial-in' or 'dial-out'
            server_template_name = (
                f"server/server_twilio_daily_sip_{mode.replace('-', '')}.py.jinja2"
            )
            utils_template_name = (
                f"server/server_utils_twilio_daily_sip_{mode.replace('-', '')}.py.jinja2"
            )
        else:
            # Shouldn't happen, but provide a fallback
            return

        # Generate server.py
        server_template = self.env.get_template(server_template_name)
        server_content = server_template.render()
        (project_path / "server.py").write_text(server_content, encoding="utf-8")

        # Generate server_utils.py
        utils_template = self.env.get_template(utils_template_name)
        utils_content = utils_template.render()
        (project_path / "server_utils.py").write_text(utils_content, encoding="utf-8")

    def _needs_aiohttp_session(self) -> bool:
        """Check if any selected service requires an aiohttp session."""
        # Collect all selected services
        service_values = []

        if self.config.mode == "cascade":
            if self.config.stt_service:
                service_values.append(self.config.stt_service)
            if self.config.llm_service:
                service_values.append(self.config.llm_service)
            if self.config.tts_service:
                service_values.append(self.config.tts_service)
        else:
            if self.config.realtime_service:
                service_values.append(self.config.realtime_service)

        if self.config.video_service:
            service_values.append(self.config.video_service)

        # Check if any service config contains "session=" parameter
        for service_value in service_values:
            if service_value in ServiceRegistry.SERVICE_CONFIGS:
                config_str = ServiceRegistry.SERVICE_CONFIGS[service_value]
                # Check if the config contains session= or aiohttp_session=
                if "session=" in config_str or "aiohttp_session=" in config_str:
                    return True

        return False

    def _generate_bot_file(self, project_path: Path) -> None:
        """Generate the main bot.py file."""
        # Select template based on mode
        if self.config.mode == "cascade":
            template = self.env.get_template("server/bot_cascade.py.jinja2")
        else:
            template = self.env.get_template("server/bot_realtime.py.jinja2")

        # Prepare context for template
        services: dict[str, str | list[str]] = {
            "transports": self.config.transports,
        }

        if self.config.mode == "cascade":
            for key, value in (
                ("stt", self.config.stt_service),
                ("llm", self.config.llm_service),
                ("tts", self.config.tts_service),
            ):
                if value:
                    services[key] = value
        elif self.config.realtime_service:
            services["realtime"] = self.config.realtime_service

        # Add video service if present
        if self.config.video_service:
            services["video"] = self.config.video_service

        features = {
            "recording": self.config.recording,
            "transcription": self.config.transcription,
            "observability": self.config.enable_observability,
            "eval": self.config.enable_eval,
        }

        # Get imports
        imports = ServiceLoader.get_imports_for_services(services, features, self.config.bot_type)

        # Check if we need aiohttp session and add import if needed
        needs_session = self._needs_aiohttp_session()
        if needs_session and "import aiohttp" not in imports:
            imports.insert(0, "import aiohttp")

        context = {
            "project_name": self.config.project_name,
            "imports": imports,
            "bot_type": self.config.bot_type,
            "transports": self.config.transports,
            "mode": self.config.mode,
            "stt_service": self.config.stt_service,
            "external_turn_detection": ServiceLoader.uses_external_turn_detection(
                self.config.stt_service
            ),
            "llm_service": self.config.llm_service,
            "tts_service": self.config.tts_service,
            "realtime_service": self.config.realtime_service,
            "video_service": self.config.video_service,
            "video_input": self.config.video_input,
            "video_output": self.config.video_output,
            "recording": self.config.recording,
            "transcription": self.config.transcription,
            "enable_krisp": self.config.enable_krisp,
            "enable_observability": self.config.enable_observability,
            "enable_eval": self.config.enable_eval,
            "service_configs": ServiceRegistry.SERVICE_CONFIGS,
            "daily_pstn_mode": self.config.daily_pstn_mode,
            "twilio_daily_sip_mode": self.config.twilio_daily_sip_mode,
            "needs_session": needs_session,
        }

        # Render and write
        content = template.render(**context)
        bot_file = project_path / "bot.py"
        bot_file.write_text(content, encoding="utf-8")

    def _generate_pyproject(self, project_path: Path) -> None:
        """Generate pyproject.toml with dependencies."""
        template = self.env.get_template("server/pyproject.toml.jinja2")

        # Build pipecat-ai extras list using ServiceLoader
        services: dict[str, str | list[str]] = {"transports": self.config.transports}

        if self.config.mode == "cascade":
            for key, value in (
                ("stt", self.config.stt_service),
                ("llm", self.config.llm_service),
                ("tts", self.config.tts_service),
            ):
                if value:
                    services[key] = value
        elif self.config.realtime_service:
            services["realtime"] = self.config.realtime_service

        # Extract all required extras
        extras = ServiceLoader.extract_extras_for_services(services)

        # The `evals` extra bundles the `pipecat eval` command and the harness's
        # local speech stack (Kokoro + Moonshine) to run the starter scenarios.
        if self.config.enable_eval:
            extras.add("evals")

        # Build the pipecat-ai dependency string. Floor at 1.4.0: generated bots use
        # create_transport + the typed CallData/runner-args API, which land in 1.4.0.
        pipecat_extras = ",".join(sorted(extras))
        pipecat_dependency = f"pipecat-ai[{pipecat_extras}]>=1.4.0"

        context = {
            "project_name": self.config.project_name,
            "pipecat_dependency": pipecat_dependency,
            "deploy_to_cloud": self.config.deploy_to_cloud,
            "enable_observability": self.config.enable_observability,
            "transports": self.config.transports,
        }

        content = template.render(**context)
        (project_path / "pyproject.toml").write_text(content, encoding="utf-8")

    def _generate_env_example(self, project_path: Path) -> None:
        """Generate .env.example with required API keys."""
        template = self.env.get_template("server/env.example.jinja2")

        context = {
            "project_name": self.config.project_name,
            "transports": self.config.transports,
            "stt_service": self.config.stt_service,
            "llm_service": self.config.llm_service,
            "tts_service": self.config.tts_service,
            "realtime_service": self.config.realtime_service,
            "video_service": self.config.video_service,
            "daily_pstn_mode": self.config.daily_pstn_mode,
            "twilio_daily_sip_mode": self.config.twilio_daily_sip_mode,
        }

        content = template.render(**context)
        (project_path / ".env.example").write_text(content, encoding="utf-8")

    def _generate_eval_scenarios(self, server_path: Path) -> None:
        """Generate starter eval scenarios in server/evals/.

        Two runnable scenarios that pass against the freshly scaffolded bot and
        double as local schema references to copy when adding more: a text-mode
        one (the fast inner loop) and an audio-mode one (the full round trip).
        A realtime (speech-to-speech) bot has no separate text LLM step, so it
        gets only the audio scenario.
        """
        evals_path = server_path / "evals"
        evals_path.mkdir(exist_ok=True)

        scenarios = ["starter_audio.yaml"]
        if self.config.mode == "cascade":
            scenarios.insert(0, "starter_text.yaml")

        context = {
            "project_name": self.config.project_name,
            "mode": self.config.mode,
        }
        for scenario in scenarios:
            template = self.env.get_template(f"server/evals/{scenario}.jinja2")
            content = template.render(**context)
            (evals_path / scenario).write_text(content, encoding="utf-8")

    def _generate_gitignore(self, project_path: Path) -> None:
        """Generate .gitignore file."""
        template = self.env.get_template("gitignore.jinja2")
        context = {
            "generate_client": self.config.generate_client,
        }
        content = template.render(**context)
        (project_path / ".gitignore").write_text(content, encoding="utf-8")

    def _get_service_label(self, service_value: str | None, service_list: list) -> str | None:
        """Get human-readable label for a service value."""
        if not service_value:
            return None
        return next(
            (svc.label for svc in service_list if svc.value == service_value), service_value
        )

    def _generate_readme(self, project_path: Path) -> None:
        """Generate README.md with project-specific instructions."""
        template = self.env.get_template("README.md.jinja2")

        # Get human-readable labels for all services
        all_transports = ServiceRegistry.WEBRTC_TRANSPORTS + ServiceRegistry.TELEPHONY_TRANSPORTS

        # Categorize transports for the run instructions
        telephony_transports = {"twilio", "telnyx", "plivo", "exotel"}
        webrtc_transports = {"smallwebrtc", "daily"}
        has_telephony = any(t in telephony_transports for t in self.config.transports)
        has_webrtc = any(t in webrtc_transports for t in self.config.transports)

        context = {
            "project_name": self.config.project_name,
            "bot_type": self.config.bot_type,
            "transports": self.config.transports,
            "transport_labels": [
                self._get_service_label(t, all_transports) for t in self.config.transports
            ],
            "mode": self.config.mode,
            "stt_service": self.config.stt_service,
            "stt_label": self._get_service_label(
                self.config.stt_service, ServiceRegistry.STT_SERVICES
            ),
            "llm_service": self.config.llm_service,
            "llm_label": self._get_service_label(
                self.config.llm_service, ServiceRegistry.LLM_SERVICES
            ),
            "tts_service": self.config.tts_service,
            "tts_label": self._get_service_label(
                self.config.tts_service, ServiceRegistry.TTS_SERVICES
            ),
            "realtime_service": self.config.realtime_service,
            "realtime_label": self._get_service_label(
                self.config.realtime_service, ServiceRegistry.REALTIME_SERVICES
            ),
            "video_input": self.config.video_input,
            "video_output": self.config.video_output,
            "recording": self.config.recording,
            "transcription": self.config.transcription,
            "enable_krisp": self.config.enable_krisp,
            "enable_observability": self.config.enable_observability,
            "enable_eval": self.config.enable_eval,
            "deploy_to_cloud": self.config.deploy_to_cloud,
            "generate_client": self.config.generate_client,
            "client_framework": self.config.client_framework,
            "client_server": self.config.client_server,
            "has_telephony": has_telephony,
            "has_webrtc": has_webrtc,
            "daily_pstn_mode": self.config.daily_pstn_mode,
            "twilio_daily_sip_mode": self.config.twilio_daily_sip_mode,
        }

        content = template.render(**context)
        (project_path / "README.md").write_text(content, encoding="utf-8")

    def _generate_dockerfile(self, project_path: Path) -> None:
        """Generate Dockerfile for Pipecat Cloud deployment."""
        template = self.env.get_template("server/Dockerfile.jinja2")

        context = {
            "transports": self.config.transports,
            "daily_pstn_mode": self.config.daily_pstn_mode,
            "twilio_daily_sip_mode": self.config.twilio_daily_sip_mode,
        }

        content = template.render(**context)
        (project_path / "Dockerfile").write_text(content, encoding="utf-8")

    def _generate_pcc_deploy(self, project_path: Path) -> None:
        """Generate pcc-deploy.toml for Pipecat Cloud deployment."""
        template = self.env.get_template("server/pcc-deploy.toml.jinja2")

        context = {
            "project_name": self.config.project_name,
            "enable_krisp": self.config.enable_krisp,
        }

        content = template.render(**context)
        (project_path / "pcc-deploy.toml").write_text(content, encoding="utf-8")

    def print_next_steps(self, project_path: Path, in_place: bool = False) -> None:
        """Print next steps for the user.

        Args:
            project_path: The directory the project was created in.
            in_place: If True, the project was scaffolded into the current directory,
                so the "cd into your project" step is omitted.
        """
        console.print("\n[bold green]✨ Project created successfully![/bold green]")
        console.print(f"   [cyan]{project_path}[/cyan]\n")

        console.print("[bold]Next steps:[/bold]\n")
        if not in_place:
            console.print(
                f"  • Go to your project: [bold cyan]cd {self.config.project_name}[/bold cyan]"
            )

        # Check if this is Daily PSTN or Twilio + Daily SIP (special handling)
        is_daily_pstn = any(
            t in ["daily_pstn_dialin", "daily_pstn_dialout"] for t in self.config.transports
        )
        is_twilio_daily_sip = any(
            t in ["twilio_daily_sip_dialin", "twilio_daily_sip_dialout"]
            for t in self.config.transports
        )

        if is_daily_pstn or is_twilio_daily_sip:
            # Special instructions for Daily PSTN and Twilio + Daily SIP
            console.print("\n  [bold]Server setup:[/bold]")
            console.print("  • Go to server: [bold cyan]cd server[/bold cyan]")
            console.print("  • Install dependencies: [bold cyan]uv sync[/bold cyan]")
            console.print("  • Create .env file: [bold cyan]cp .env.example .env[/bold cyan]")
            console.print("  • [bold]Edit .env and add your API keys[/bold]")
            console.print("\n  [bold]See README.md for detailed setup instructions[/bold]")
            console.print("  • Configure webhooks/SIP domains as described in the README")
            console.print("  • Run the multi-terminal workflow (server.py + bot.py)")

            if self.config.deploy_to_cloud:
                console.print(
                    "\n[dim]See https://docs.pipecat.ai/deployment/pipecat-cloud for deployment info.[/dim]\n"
                )
            else:
                console.print(
                    "\n[dim]Check the README for local development and production deployment.[/dim]\n"
                )
            return

        # Client setup
        if self.config.generate_client:
            console.print("\n  [bold]Client setup:[/bold]")
            console.print(
                "  • Go to client: In a separate terminal window or tab [bold cyan]cd client[/bold cyan]"
            )
            console.print("  • Install dependencies: [bold cyan]npm install[/bold cyan]")
            console.print("  • Run dev server: [bold cyan]npm run dev[/bold cyan]")

        # Server setup
        console.print("\n  [bold]Server setup:[/bold]")
        console.print("  • Go to server: [bold cyan]cd server[/bold cyan]")
        console.print("  • Install dependencies: [bold cyan]uv sync[/bold cyan]")
        console.print("  • Create .env file: [bold cyan]cp .env.example .env[/bold cyan]")
        console.print("  • [bold]Edit .env and add your API keys[/bold]")

        # Every standard transport runs the same way: `uv run bot.py`. The runner
        # serves all transports and the caller selects which one — a web/mobile client
        # picks its transport when it connects, and a telephony provider connects to
        # /ws. Telephony additionally needs a public tunnel so the provider can reach
        # the bot.
        telephony_transports = {"twilio", "telnyx", "plivo", "exotel"}
        has_telephony = any(t in telephony_transports for t in self.config.transports)

        console.print("  • Run your bot: [bold cyan]uv run bot.py[/bold cyan]")
        if self.config.enable_eval:
            starter = (
                "evals/starter_text.yaml"
                if self.config.mode == "cascade"
                else "evals/starter_audio.yaml"
            )
            console.print(
                "  • Test it headless: [bold cyan]uv run bot.py -t eval[/bold cyan], then in "
                f"another terminal [bold cyan]uv run pipecat eval run {starter} -v[/bold cyan]"
            )
        if has_telephony:
            console.print(
                "  • Expose it for telephony: [bold cyan]ngrok http 7860[/bold cyan], then point "
                "your provider's webhook at [bold cyan]wss://<your-ngrok-host>/ws[/bold cyan]"
            )

        # Add cloud deployment info if applicable
        if self.config.deploy_to_cloud:
            console.print(
                "\n[dim]See https://docs.pipecat.ai/deployment/pipecat-cloud for deployment info.[/dim]\n"
            )
        else:
            console.print("\n[dim]See README.md for detailed setup instructions.[/dim]\n")

    def _generate_client(self, client_path: Path) -> None:
        """Generate client application files."""
        # Determine which template to use
        # Determine template directory based on framework and server
        if self.config.client_framework == "react":
            if self.config.client_server == "vite":
                template_dir = "client/react-vite"
            elif self.config.client_server == "nextjs":
                template_dir = "client/react-nextjs"
            else:
                console.print(
                    f"[yellow]⚠️  Unknown client server: {self.config.client_server}[/yellow]"
                )
                return
        elif self.config.client_framework == "vanilla":
            # Vanilla JS always uses Vite
            template_dir = "client/vanilla-js-vite"
        else:
            console.print(
                f"[yellow]⚠️  Unknown client framework: {self.config.client_framework}[/yellow]"
            )
            return

        # Get the template directory path
        import pipecat.cli

        package_path = Path(pipecat.cli.__file__).parent
        source_template_dir = package_path / "templates" / template_dir

        if not source_template_dir.exists():
            console.print(f"[yellow]⚠️  Template not found: {source_template_dir}[/yellow]")
            return

        # Copy all files and render .jinja2 templates
        self._copy_and_render_directory(source_template_dir, client_path)

    def _copy_and_render_directory(self, source_dir: Path, dest_dir: Path) -> None:
        """Recursively copy directory contents, rendering .jinja2 templates.

        Args:
            source_dir: Source template directory
            dest_dir: Destination directory
        """
        for item in source_dir.rglob("*"):
            if item.is_file():
                # Calculate relative path
                rel_path = item.relative_to(source_dir)

                # Skip SmallWebRTC-specific files if SmallWebRTC is not in transports
                if "smallwebrtc" not in self.config.transports:
                    # Skip the sessions API route for Next.js
                    if "api/sessions" in str(rel_path):
                        continue

                # Determine destination path
                if item.suffix == ".jinja2":
                    # Remove .jinja2 extension for rendered files
                    dest_file = dest_dir / str(rel_path)[: -len(".jinja2")]
                else:
                    dest_file = dest_dir / rel_path

                # Create parent directories
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                # Render or copy
                if item.suffix == ".jinja2":
                    self._render_client_template(item, dest_file)
                else:
                    shutil.copy2(item, dest_file)

    def _render_client_template(self, template_file: Path, dest_file: Path) -> None:
        """Render a Jinja2 template file with client-specific context.

        Only used for config.ts and package.json templates.
        Most TypeScript files are static and copied directly.

        Args:
            template_file: Path to .jinja2 template
            dest_file: Destination file path (without .jinja2)
        """
        # Create Jinja2 environment for this specific file
        template_content = template_file.read_text(encoding="utf-8")
        from jinja2 import Template

        try:
            template = Template(template_content)
        except Exception as e:
            console.print(f"[red]Error rendering template {template_file.name}:[/red]")
            console.print(f"[red]{e}[/red]")
            raise

        # Prepare context - only need transport values and project name
        # Transform transport strings to objects for template iteration
        transport_objects = [{"value": t} for t in self.config.transports]

        context = {
            "project_name": self.config.project_name,
            "transports": transport_objects,
        }

        # Render and write
        rendered = template.render(**context)
        dest_file.write_text(rendered, encoding="utf-8")

    def _ruff_command(self) -> list[str]:
        """Resolve the command used to invoke Ruff.

        Ruff is a hard dependency of the CLI, so it is always installed in the
        same environment. We invoke the bundled binary directly (rather than a
        bare ``ruff`` on PATH), because when the CLI is installed as an isolated
        tool (``uv tool install`` / ``pipx``) only the ``pipecat``/``pc`` scripts
        are exposed on PATH — Ruff's console script is not.
        """
        try:
            from ruff.__main__ import find_ruff_bin

            return [find_ruff_bin()]
        except (ImportError, FileNotFoundError):
            # Fall back to invoking Ruff as a module via the current interpreter.
            return [sys.executable, "-m", "ruff"]

    def _format_python_files(self, project_path: Path) -> None:
        """Format generated Python files with Ruff."""
        ruff = self._ruff_command()
        try:
            # Run ruff format on the project directory
            fmt = subprocess.run(
                [*ruff, "format", str(project_path)],
                capture_output=True,
                text=True,
                check=False,
            )

            # Run ruff check --fix to organize imports
            imports = subprocess.run(
                [*ruff, "check", "--fix", "--select", "I", str(project_path)],
                capture_output=True,
                text=True,
                check=False,
            )
        except (FileNotFoundError, OSError) as e:
            console.print(
                f"[yellow]⚠️  Could not run Ruff to format generated files ({e}). "
                "Generated Python may be unformatted; run 'ruff format' manually.[/yellow]"
            )
            return

        # Surface non-zero exits instead of silently shipping unformatted code.
        for label, result in (("format", fmt), ("check --fix", imports)):
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                console.print(
                    f"[yellow]⚠️  Ruff {label} exited with code {result.returncode}; "
                    "generated Python may be unformatted."
                    + (f"\n{detail}" if detail else "")
                    + "[/yellow]"
                )
