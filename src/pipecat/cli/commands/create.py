#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Create command implementation for scaffolding new Pipecat projects."""

import json
from pathlib import Path

import typer
from rich.console import Console

from pipecat.cli.generators import ProjectGenerator
from pipecat.cli.prompts import ProjectConfig, ask_project_questions
from pipecat.cli.registry.service_metadata import ServiceRegistry

console = Console()


def _list_options_callback(value: bool):
    """Print available options as JSON and exit."""
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


def create_command(
    ctx: typer.Context,
    target: str | None = typer.Argument(
        None,
        help="Target directory; use '.' to scaffold into the current directory "
        "(no <name> subfolder). Omit to create a <name> subfolder.",
    ),
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory (defaults to current directory)"
    ),
    list_options: bool = typer.Option(
        False,
        "--list-options",
        help="Print available service options as JSON and exit",
        callback=_list_options_callback,
        is_eager=True,
    ),
    # --- Non-interactive flags ---
    name: str | None = typer.Option(
        None, "--name", "-n", help="Project name (triggers non-interactive mode)"
    ),
    bot_type: str | None = typer.Option(
        None,
        "--bot-type",
        "-b",
        help="Bot type: 'web' or 'telephony' (inferred from --transport if omitted)",
    ),
    transport: list[str] | None = typer.Option(
        None, "--transport", "-t", help="Transport (repeatable, e.g. -t daily -t smallwebrtc)"
    ),
    mode: str | None = typer.Option(
        None, "--mode", "-m", help="Pipeline mode: 'cascade' or 'realtime'"
    ),
    stt: str | None = typer.Option(None, "--stt", help="STT service (cascade mode)"),
    llm: str | None = typer.Option(None, "--llm", help="LLM service (cascade mode)"),
    tts: str | None = typer.Option(None, "--tts", help="TTS service (cascade mode)"),
    realtime: str | None = typer.Option(
        None, "--realtime", help="Realtime service (realtime mode)"
    ),
    video: str | None = typer.Option(None, "--video", help="Video avatar service"),
    client_framework: str | None = typer.Option(
        None, "--client-framework", help="Client framework: 'react', 'vanilla', or 'none'"
    ),
    client_server: str | None = typer.Option(
        None, "--client-server", help="Client dev server: 'vite' or 'nextjs'"
    ),
    daily_pstn_mode: str | None = typer.Option(
        None, "--daily-pstn-mode", help="Daily PSTN mode: 'dial-in' or 'dial-out'"
    ),
    twilio_daily_sip_mode: str | None = typer.Option(
        None, "--twilio-daily-sip-mode", help="Twilio+Daily SIP mode: 'dial-in' or 'dial-out'"
    ),
    recording: bool = typer.Option(False, "--recording/--no-recording", help="Enable recording"),
    transcription: bool = typer.Option(
        False, "--transcription/--no-transcription", help="Enable transcription"
    ),
    video_input: bool = typer.Option(
        False, "--video-input/--no-video-input", help="Enable video input"
    ),
    video_output: bool = typer.Option(
        False, "--video-output/--no-video-output", help="Enable video output"
    ),
    deploy_to_cloud: bool = typer.Option(
        True, "--deploy-to-cloud/--no-deploy-to-cloud", help="Generate cloud deployment files"
    ),
    enable_krisp: bool = typer.Option(
        False, "--enable-krisp/--no-enable-krisp", help="Enable Krisp noise cancellation"
    ),
    observability: bool = typer.Option(
        False, "--observability/--no-observability", help="Enable observability"
    ),
    enable_eval: bool = typer.Option(
        False,
        "--eval/--no-eval",
        help="Add an 'eval' transport so the bot is runnable with `-t eval` for "
        "behavioral evals (see `pipecat eval`). Off by default.",
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="JSON config file (triggers non-interactive mode)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print resolved config as JSON without generating files"
    ),
):
    r"""Create a new Pipecat project.

    Creates a complete project structure with bot.py, dependencies, and configuration files.

    In interactive mode (default), uses a wizard to guide you through setup.
    In non-interactive mode (when --name or --config is provided), all configuration
    is taken from flags or a config file.

    Examples::

        pc create                                          # Interactive wizard
        pc create .                                        # Scaffold into the current dir
        pc create --name my-bot --bot-type web \
          --transport daily --mode cascade \
          --stt deepgram_stt --llm openai_llm \
          --tts cartesia_tts                             # Non-interactive
        pc create . --bot-type web -t daily -m cascade \
          --stt deepgram_stt --llm openai_llm \
          --tts cartesia_tts                             # In-place, name from dir
        pc create --config project-config.json             # From config file
        pc create --name my-bot ... --dry-run              # Preview config as JSON
    """
    # `quickstart` is dispatched here rather than as a subcommand: a positional arg on a
    # Typer *group* can't be followed by options (Click stops parsing at the first
    # positional), which would break `pc create . --bot-type ...`. Keeping `create` a plain
    # command and routing the `quickstart` token preserves `pc create quickstart [-o ...]`.
    #
    # NOTE: this is the `pipecat create` scaffolder (formerly `pipecat init`). `pipecat init`
    # is now a separate command that makes a project agent-ready (see commands/init.py).
    if target == "quickstart":
        return quickstart_command(output_dir=output_dir)

    try:
        # Resolve the in-place target. A positional path is the exact destination
        # (vite/django style); -o keeps its legacy "parent dir + name subfolder" meaning,
        # so passing both is ambiguous.
        in_place = target is not None
        dest: Path | None = None
        derived_name: str | None = None
        if in_place:
            if output_dir is not None:
                console.print(
                    "\n[red]Error: pass either a target directory or --output, not both.[/red]"
                )
                raise typer.Exit(1)
            dest = Path(target).resolve()
            derived_name = dest.name or "pipecat-app"

        # In-place runs go non-interactive as soon as any non-interactive intent is
        # present (the name can be derived from the directory). Scoped to in_place so
        # no existing (no-positional) invocation changes behavior.
        non_interactive = name is not None or config is not None
        if in_place and (name is not None or bot_type is not None or config is not None):
            non_interactive = True

        if non_interactive:
            from pipecat.cli.config_validator import (
                ConfigValidationError,
                config_to_json,
                load_config_from_file,
                validate_and_build_config,
            )

            # Load from config file if provided
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

            # Generate project
            generator = ProjectGenerator(project_config)
            project_path = generator.generate(
                dest if in_place else output_dir, non_interactive=True, in_place=in_place
            )

            # Show next steps
            generator.print_next_steps(project_path, in_place=in_place)

        else:
            # Interactive mode: ask questions
            config_result = ask_project_questions(default_name=derived_name)

            # Generate project
            generator = ProjectGenerator(config_result)
            project_path = generator.generate(dest if in_place else output_dir, in_place=in_place)

            # Show next steps
            generator.print_next_steps(project_path, in_place=in_place)

    except KeyboardInterrupt:
        console.print("\n[yellow]Project creation cancelled.[/yellow]")
        raise typer.Exit(1)
    except typer.Exit:
        raise
    except FileExistsError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error creating project: {e}[/red]")
        raise typer.Exit(1)


def quickstart_command(output_dir: Path | None = None):
    """Create a new Pipecat project with quickstart defaults.

    Sets up a project with SmallWebRTC, Deepgram STT, OpenAI LLM, and Cartesia TTS
    — the fastest way to get a voice agent running. Dispatched from ``create_command``
    when the target is ``quickstart`` (e.g. ``pc create quickstart [-o DIR]``).
    """
    try:
        project_name = "pipecat-quickstart"

        console.print("[bold cyan]Let's create your Pipecat project![/bold cyan]\n")

        # Display all pre-selected defaults
        console.print(f"[green]✔[/green] Project name: [cyan]{project_name}[/cyan]")
        console.print("[green]✔[/green] Bot type: [cyan]Web/Mobile[/cyan]")
        console.print("[green]✔[/green] Transport: [cyan]SmallWebRTC, Daily[/cyan]")
        console.print(
            "[green]✔[/green] Pipeline architecture: [cyan]Cascade (STT → LLM → TTS)[/cyan]"
        )
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
        project_path = generator.generate(output_dir, non_interactive=True)

        # Show next steps
        generator.print_next_steps(project_path)

    except KeyboardInterrupt:
        console.print("\n[yellow]Project creation cancelled.[/yellow]")
        raise typer.Exit(1)
    except typer.Exit:
        raise
    except FileExistsError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error creating project: {e}[/red]")
        raise typer.Exit(1)
