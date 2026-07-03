#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Project scaffolding for ``pipecat init``.

The scaffolder generates a runnable Pipecat project (``bot.py``, dependencies, config,
optional client). It has three entry points, all used by ``pipecat init``:

- :func:`scaffold_interactive` — the wizard, run when the developer chooses "scaffold a
  runnable bot now" on interactive ``init``.
- :func:`scaffold_quickstart` — the canned quickstart preset (``pipecat init quickstart``).
- :func:`resolve_scaffold_config` + :func:`generate_scaffold` — flags/config-file driven,
  no prompts (``pipecat init . --bot-type web …``); the path coding agents and automation
  use. Split so the config is validated before any files are written.

These were formerly the body of the standalone ``pipecat create`` command, which has been
removed — ``pipecat init`` is now the single entry point.
"""

import json
from pathlib import Path

import typer
from rich.console import Console

from pipecat.cli.generators import ProjectGenerator
from pipecat.cli.prompts import ProjectConfig, ask_project_questions
from pipecat.cli.registry.service_metadata import ServiceRegistry

console = Console()


def list_options_callback(value: bool):
    """Print available service options as JSON and exit (eager ``--list-options`` flag)."""
    if not value:
        return

    def values(defs):
        return [s.value for s in defs]

    options = {
        "bot_type": ["web", "telephony"],
        "transports": {
            "web": values(ServiceRegistry.WEBRTC_TRANSPORTS),
            "telephony": values(ServiceRegistry.TELEPHONY_TRANSPORTS),
        },
        "stt": values(ServiceRegistry.STT_SERVICES),
        "llm": values(ServiceRegistry.LLM_SERVICES),
        "tts": values(ServiceRegistry.TTS_SERVICES),
        "realtime": values(ServiceRegistry.REALTIME_SERVICES),
        "video": values(ServiceRegistry.VIDEO_SERVICES),
    }
    print(json.dumps(options, indent=2))
    raise typer.Exit(0)


def scaffold_interactive(dest: Path | None, derived_name: str | None, in_place: bool):
    """Run the interactive wizard and generate a project.

    Used by ``pipecat init`` when the developer chooses to scaffold a runnable bot.
    ``dest`` is the resolved output location: the exact target directory when ``in_place``
    is True, otherwise the parent the ``<name>`` subfolder is created under (``None`` =
    current directory).
    """
    config_result = ask_project_questions(default_name=derived_name)
    generator = ProjectGenerator(config_result)
    project_path = generator.generate(dest, in_place=in_place)
    generator.print_next_steps(project_path, in_place=in_place)


def resolve_scaffold_config(
    ctx: typer.Context,
    *,
    derived_name: str | None,
    dry_run: bool,
    config: Path | None,
    name: str | None,
    bot_type: str | None,
    transport: list[str] | None,
    mode: str | None,
    stt: str | None,
    llm: str | None,
    tts: str | None,
    realtime: str | None,
    video: str | None,
    client_framework: str | None,
    client_server: str | None,
    daily_pstn_mode: str | None,
    twilio_daily_sip_mode: str | None,
    recording: bool,
    transcription: bool,
    video_input: bool,
    video_output: bool,
    deploy_to_cloud: bool,
    enable_krisp: bool,
    observability: bool,
    enable_eval: bool,
):
    """Merge flags + config file into a validated ``ProjectConfig`` — no file writes.

    Merges a ``--config`` file (if given) with the CLI flags and validates the result. An
    explicit CLI flag always wins; the file value applies only when the flag was omitted.
    Exits non-zero with a clear message on a validation error, and on ``dry_run`` prints
    the resolved config as JSON and exits zero — so callers can validate *before* writing
    anything (see :func:`generate_scaffold`).

    ``ctx`` is the calling Typer command's context — used to tell which flags the user
    actually typed (vs. their defaults). The parameter names here must match the option
    names declared on that command.
    """
    from pipecat.cli.config_validator import (
        ConfigValidationError,
        config_to_json,
        load_config_from_file,
        validate_and_build_config,
    )

    if config is not None:
        file_data = load_config_from_file(config)

        # Merge file values with CLI flags. An explicit CLI flag always wins;
        # the file value only applies when the flag was omitted.
        def from_cli(param):
            """True only if the user typed this flag (vs. falling back to its default).

            Compares the parameter source by enum member name rather than identity:
            Typer vendors its own copy of Click, so the ``ParameterSource`` returned
            here is a different enum object than ``click.core.ParameterSource``.
            """
            source = ctx.get_parameter_source(param)
            return source is not None and source.name == "COMMANDLINE"

        def pick(value, param, *file_keys):
            """Explicit CLI flag wins; else first file key present; else the flag's default."""
            if from_cli(param):
                return value
            for key in file_keys:
                if key in file_data:
                    return file_data[key]
            return value

        name = pick(name, "name", "name", "project_name")
        bot_type = pick(bot_type, "bot_type", "bot_type")
        transport = pick(transport, "transport", "transports", "transport")
        mode = pick(mode, "mode", "mode")
        stt = pick(stt, "stt", "stt", "stt_service")
        llm = pick(llm, "llm", "llm", "llm_service")
        tts = pick(tts, "tts", "tts", "tts_service")
        realtime = pick(realtime, "realtime", "realtime", "realtime_service")
        video = pick(video, "video", "video", "video_service")
        client_framework = pick(client_framework, "client_framework", "client_framework")
        client_server = pick(client_server, "client_server", "client_server")
        daily_pstn_mode = pick(daily_pstn_mode, "daily_pstn_mode", "daily_pstn_mode")
        twilio_daily_sip_mode = pick(
            twilio_daily_sip_mode, "twilio_daily_sip_mode", "twilio_daily_sip_mode"
        )
        recording = pick(recording, "recording", "recording")
        transcription = pick(transcription, "transcription", "transcription")
        video_input = pick(video_input, "video_input", "video_input")
        video_output = pick(video_output, "video_output", "video_output")
        deploy_to_cloud = pick(deploy_to_cloud, "deploy_to_cloud", "deploy_to_cloud")
        enable_krisp = pick(enable_krisp, "enable_krisp", "enable_krisp")
        observability = pick(
            observability, "observability", "observability", "enable_observability"
        )
        enable_eval = pick(enable_eval, "enable_eval", "enable_eval", "eval")

    try:
        project_config = validate_and_build_config(
            name=name or derived_name,
            bot_type=bot_type,
            transport=transport,
            mode=mode,
            stt=stt,
            llm=llm,
            tts=tts,
            realtime=realtime,
            video=video,
            client_framework=client_framework,
            client_server=client_server,
            daily_pstn_mode=daily_pstn_mode,
            twilio_daily_sip_mode=twilio_daily_sip_mode,
            recording=recording,
            transcription=transcription,
            video_input=video_input,
            video_output=video_output,
            deploy_to_cloud=deploy_to_cloud,
            enable_krisp=enable_krisp,
            observability=observability,
            enable_eval=enable_eval,
        )
    except ConfigValidationError as e:
        console.print(f"\n[red]{e}[/red]")
        raise typer.Exit(1)

    if dry_run:
        print(config_to_json(project_config))
        raise typer.Exit(0)

    return project_config


def generate_scaffold(project_config, *, dest: Path | None, in_place: bool):
    """Generate a project from an already-validated config and print next steps.

    The write half of non-interactive scaffolding; pair it with
    :func:`resolve_scaffold_config`, which validates first so a bad invocation fails
    without touching the directory.
    """
    generator = ProjectGenerator(project_config)
    project_path = generator.generate(dest, non_interactive=True, in_place=in_place)
    generator.print_next_steps(project_path, in_place=in_place)


def scaffold_quickstart(
    output_dir: Path | None = None, *, dest: Path | None = None, in_place: bool = False
):
    """Generate the canned quickstart project (no questions).

    Sets up a project with SmallWebRTC, Daily, Deepgram STT, OpenAI LLM, and Cartesia
    TTS — the fastest way to get a voice agent running. Used by ``pipecat init
    quickstart``, which scaffolds in-place into an already-initialized directory (via
    ``dest`` / ``in_place``).
    """
    project_name = "pipecat-quickstart"

    console.print("[bold cyan]Let's create your Pipecat project![/bold cyan]\n")

    # Display all pre-selected defaults
    console.print(f"[green]✔[/green] Project name: [cyan]{project_name}[/cyan]")
    console.print("[green]✔[/green] Bot type: [cyan]Web/Mobile[/cyan]")
    console.print("[green]✔[/green] Transport: [cyan]SmallWebRTC, Daily[/cyan]")
    console.print("[green]✔[/green] Pipeline architecture: [cyan]Cascade (STT → LLM → TTS)[/cyan]")
    console.print("[green]✔[/green] Speech-to-Text: [cyan]Deepgram[/cyan]")
    console.print("[green]✔[/green] Language model: [cyan]OpenAI[/cyan]")
    console.print("[green]✔[/green] Text-to-Speech: [cyan]Cartesia[/cyan]")
    console.print("[green]✔[/green] Deploy to Pipecat Cloud: [cyan]Yes[/cyan]")

    # Build config with quickstart defaults
    project_config = ProjectConfig(
        project_name=project_name,
        bot_type="web",
        transports=["smallwebrtc", "daily"],
        mode="cascade",
        stt_service="deepgram_stt",
        llm_service="openai_responses_llm",
        tts_service="cartesia_tts",
        deploy_to_cloud=True,
    )

    # Generate project
    generator = ProjectGenerator(project_config)
    project_path = generator.generate(
        dest if in_place else output_dir, non_interactive=True, in_place=in_place
    )

    # Show next steps
    generator.print_next_steps(project_path, in_place=in_place)
