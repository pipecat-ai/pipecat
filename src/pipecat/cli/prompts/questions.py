#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Interactive prompts and question flow for project configuration."""

import sys
from dataclasses import dataclass, field

import questionary
from questionary import Choice, Style
from rich.console import Console

from pipecat.cli.registry import BotType, ServiceLoader, ServiceRegistry

console = Console()

# Custom style for cleaner, more minimal prompts (inspired by Vite)
custom_style = Style(
    [
        ("qmark", "fg:#00d7af bold"),  # Green question mark
        ("question", "bold"),  # Question text
        ("answer", "fg:#5fd7ff"),  # User's answer in cyan
        ("pointer", "fg:#00d7af bold"),  # Green pointer
        ("highlighted", "fg:#00d7af bold"),  # Selected item
        ("selected", "fg:#00d7af"),  # Selected in checkbox
        ("separator", "fg:#6c6c6c"),  # Dim separator
        ("instruction", "fg:#808080"),  # Instructions
        ("text", ""),  # Normal text
        ("disabled", "fg:#858585 italic"),  # Disabled choices
    ]
)


def replace_question_with_answer(question: str, answer: str | list[str] | None):
    """Replace the questionary output line with a checkmark version.

    Uses ANSI escape codes to move cursor up and overwrite the line.
    """
    if isinstance(answer, list):
        answer_str = ", ".join(answer)
    else:
        answer_str = answer or ""

    # ANSI escape codes:
    # \033[A = move cursor up one line
    # \033[2K = clear entire line
    # \r = carriage return to start of line
    sys.stdout.write("\033[A")  # Move up one line
    sys.stdout.write("\033[2K")  # Clear the line
    sys.stdout.write("\r")  # Go to start of line
    sys.stdout.flush()

    # Print the new line with checkmark
    console.print(f"[green]✔[/green] {question} [cyan]{answer_str}[/cyan]")


@dataclass
class ProjectConfig:
    """Configuration for a Pipecat project."""

    # Basic info
    project_name: str
    bot_type: BotType  # "web" or "telephony"

    # Transport
    transports: list[str] = field(default_factory=list)

    # Pipeline mode
    mode: str = "cascade"  # "cascade" or "realtime"

    # Services (for cascade mode)
    stt_service: str | None = None
    llm_service: str | None = None
    tts_service: str | None = None

    # Realtime service
    realtime_service: str | None = None

    # Video service (avatars, for web/mobile bots)
    video_service: str | None = None

    # Client (for web/mobile bots)
    generate_client: bool = False
    client_framework: str | None = None  # "react"
    client_server: str | None = None  # "vite" or "nextjs"

    # Daily PSTN specific
    daily_pstn_mode: str | None = None  # "dial-in" or "dial-out"

    # Twilio + Daily SIP specific
    twilio_daily_sip_mode: str | None = None  # "dial-in" or "dial-out"

    # Features
    video_input: bool = False
    video_output: bool = False
    recording: bool = False
    transcription: bool = False

    # Deployment
    deploy_to_cloud: bool = False
    enable_krisp: bool = False

    # Observability
    enable_observability: bool = False

    # Evals: add an "eval" entry to transport_params so the bot is runnable with
    # `-t eval` for behavioral evals (see `pipecat eval`). Off by default.
    enable_eval: bool = False


def ask_project_questions(default_name: str | None = None) -> ProjectConfig:
    """Ask user for project configuration through interactive prompts.

    Args:
        default_name: Optional default for the project name prompt (e.g. the basename
            of the target directory when scaffolding in place).

    Returns:
        ProjectConfig with user's selections
    """
    console.print("[bold cyan]Let's create your Pipecat project![/bold cyan]\n")

    # Question 1: Project name
    project_name = questionary.text(
        "Project name:",
        default=default_name or "",
        style=custom_style,
        validate=lambda text: len(text) > 0 or "Project name cannot be empty",
    ).ask()

    if not project_name:
        raise KeyboardInterrupt("Project creation cancelled")

    replace_question_with_answer("Project name:", project_name)

    # Question 2: Bot type
    bot_type = questionary.select(
        "Bot type:",
        choices=[
            Choice(title="Web/Mobile", value="web"),
            Choice(title="Telephony", value="telephony"),
        ],
        style=custom_style,
    ).ask()

    if not bot_type:
        raise KeyboardInterrupt("Project creation cancelled")

    replace_question_with_answer("Bot type:", "Web/Mobile" if bot_type == "web" else "Telephony")

    # Question 2b: Client framework (only for web/mobile)
    generate_client = False
    client_framework = None
    client_server = None

    if bot_type == "web":
        client_framework = questionary.select(
            "Client framework:",
            choices=[
                Choice(title="React", value="react"),
                Choice(title="Vanilla JS", value="vanilla"),
                Choice(title="None (server only)", value="none"),
            ],
            style=custom_style,
        ).ask()

        if not client_framework:
            raise KeyboardInterrupt("Project creation cancelled")

        framework_display = {
            "react": "React",
            "vanilla": "Vanilla JS",
            "none": "None (server only)",
        }
        replace_question_with_answer(
            "Client framework:", framework_display.get(client_framework, client_framework)
        )

        if client_framework == "react":
            generate_client = True
            client_server = questionary.select(
                "React dev server:",
                choices=[
                    Choice(title="Vite", value="vite"),
                    Choice(title="Next.js", value="nextjs"),
                ],
                style=custom_style,
            ).ask()

            if not client_server:
                raise KeyboardInterrupt("Project creation cancelled")

            replace_question_with_answer(
                "React dev server:", "Vite" if client_server == "vite" else "Next.js"
            )
        elif client_framework == "vanilla":
            generate_client = True
            client_server = "vite"  # Vanilla JS always uses Vite
        else:
            client_framework = None  # Set to None if user chose "none"

    # Question 3: Primary transport selection
    transport_options = ServiceLoader.get_transport_options(bot_type)

    # For Daily PSTN and Twilio + Daily SIP, show a single option and ask for mode separately
    # Filter out the mode-specific variants and show just "Daily PSTN" and "Twilio + Daily SIP"
    display_transport_options = []
    seen_daily_pstn = False
    seen_twilio_daily_sip = False
    for svc in transport_options:
        if svc.value in ["daily_pstn_dialin", "daily_pstn_dialout"]:
            if not seen_daily_pstn:
                # Create a generic "Daily PSTN" option
                display_transport_options.append(
                    type("ServiceDef", (), {"label": "Daily PSTN", "value": "daily_pstn"})()
                )
                seen_daily_pstn = True
        elif svc.value in ["twilio_daily_sip_dialin", "twilio_daily_sip_dialout"]:
            if not seen_twilio_daily_sip:
                # Create a generic "Twilio + Daily SIP" option
                display_transport_options.append(
                    type(
                        "ServiceDef",
                        (),
                        {"label": "Twilio + Daily SIP", "value": "twilio_daily_sip"},
                    )()
                )
                seen_twilio_daily_sip = True
        else:
            display_transport_options.append(svc)

    transport_choices = [
        Choice(
            title=svc.label,
            value=svc.value,
        )
        for svc in display_transport_options
    ]

    primary_transport = questionary.select(
        "Transport:",
        choices=transport_choices,
        style=custom_style,
    ).ask()

    if not primary_transport:
        raise KeyboardInterrupt("Project creation cancelled")

    # Question 3a: If Daily PSTN selected, ask for mode
    daily_pstn_mode = None
    twilio_daily_sip_mode = None

    if primary_transport == "daily_pstn":
        daily_pstn_mode = questionary.select(
            "Daily PSTN mode:",
            choices=[
                Choice(title="Dial-in (Receive calls)", value="dial-in"),
                Choice(title="Dial-out (Make calls)", value="dial-out"),
            ],
            style=custom_style,
        ).ask()

        if not daily_pstn_mode:
            raise KeyboardInterrupt("Project creation cancelled")

        mode_display = "Dial-in" if daily_pstn_mode == "dial-in" else "Dial-out"
        replace_question_with_answer("Daily PSTN mode:", mode_display)

        # Map mode to actual service value
        primary_transport = f"daily_pstn_{daily_pstn_mode.replace('-', '')}"

    elif primary_transport == "twilio_daily_sip":
        twilio_daily_sip_mode = questionary.select(
            "Twilio + Daily SIP mode:",
            choices=[
                Choice(title="Dial-in (Receive calls)", value="dial-in"),
                Choice(title="Dial-out (Make calls)", value="dial-out"),
            ],
            style=custom_style,
        ).ask()

        if not twilio_daily_sip_mode:
            raise KeyboardInterrupt("Project creation cancelled")

        mode_display = "Dial-in" if twilio_daily_sip_mode == "dial-in" else "Dial-out"
        replace_question_with_answer("Twilio + Daily SIP mode:", mode_display)

        # Map mode to actual service value
        primary_transport = f"twilio_daily_sip_{twilio_daily_sip_mode.replace('-', '')}"

    transports = [primary_transport]

    # Get label for display
    primary_label = next(
        (svc.label for svc in transport_options if svc.value == primary_transport),
        primary_transport,
    )
    replace_question_with_answer("Transport:", primary_label)

    # Question 3b: Additional transport (different for web vs telephony)
    if bot_type == "web":
        # For web bots: offer to add another transport (commonly for local testing)
        add_backup = questionary.confirm(
            "Add another transport for local testing?",
            default=False,
            style=custom_style,
        ).ask()

        replace_question_with_answer(
            "Add another transport for local testing?", "Yes" if add_backup else "No"
        )

        if add_backup:
            # Filter out the already-selected primary transport
            backup_choices = [c for c in transport_choices if c.value != primary_transport]

            if backup_choices:
                backup_transport = questionary.select(
                    "Additional transport:",
                    choices=backup_choices,
                    style=custom_style,
                ).ask()

                if backup_transport:
                    transports.append(backup_transport)
                    backup_label = next(
                        (svc.label for svc in transport_options if svc.value == backup_transport),
                        backup_transport,
                    )
                    replace_question_with_answer("Additional transport:", backup_label)

    elif bot_type == "telephony":
        # For telephony bots: offer to add WebRTC for local testing
        add_webrtc = questionary.confirm(
            "Add a WebRTC transport for local testing?",
            default=False,
            style=custom_style,
        ).ask()

        replace_question_with_answer(
            "Add a WebRTC transport for local testing?", "Yes" if add_webrtc else "No"
        )

        if add_webrtc:
            # Allows the user to choose between SmallWebRTC and Daily as a backup transport
            webrtc_choices = [
                Choice(title="SmallWebRTC", value="smallwebrtc"),
                Choice(title="Daily", value="daily"),
            ]
            webrtc_transport = questionary.select(
                "WebRTC provider:",
                choices=webrtc_choices,
                style=custom_style,
            ).ask()

            if webrtc_transport:
                transports.append(webrtc_transport)
                replace_question_with_answer(
                    "WebRTC provider:",
                    "SmallWebRTC" if webrtc_transport == "smallwebrtc" else "Daily",
                )

    # Question 4: Pipeline mode
    mode = questionary.select(
        "Pipeline architecture:",
        choices=[
            Choice(title="Cascade (STT → LLM → TTS)", value="cascade"),
            Choice(title="Realtime (speech-to-speech)", value="realtime"),
        ],
        style=custom_style,
    ).ask()

    if not mode:
        raise KeyboardInterrupt("Project creation cancelled")

    replace_question_with_answer(
        "Pipeline architecture:",
        "Cascade (STT → LLM → TTS)" if mode == "cascade" else "Realtime (speech-to-speech)",
    )

    # Initialize config
    config = ProjectConfig(
        project_name=project_name,
        bot_type=bot_type,
        transports=transports,
        mode=mode,
        generate_client=generate_client,
        client_framework=client_framework,
        client_server=client_server,
        daily_pstn_mode=daily_pstn_mode,
        twilio_daily_sip_mode=twilio_daily_sip_mode,
    )

    # Conditional questions based on mode
    if mode == "cascade":
        # Question 5a: STT Service
        stt_choices = [
            Choice(
                title=svc.label,
                value=svc.value,
            )
            for svc in ServiceRegistry.STT_SERVICES
        ]
        config.stt_service = questionary.select(
            "Speech-to-Text:",
            choices=stt_choices,
            style=custom_style,
        ).ask()

        stt_label = next(
            (svc.label for svc in ServiceRegistry.STT_SERVICES if svc.value == config.stt_service),
            config.stt_service,
        )
        replace_question_with_answer("Speech-to-Text:", stt_label)

        # Question 5b: LLM Service
        llm_choices = [
            Choice(
                title=svc.label,
                value=svc.value,
            )
            for svc in ServiceRegistry.LLM_SERVICES
        ]
        config.llm_service = questionary.select(
            "Language model:",
            choices=llm_choices,
            style=custom_style,
        ).ask()

        llm_label = next(
            (svc.label for svc in ServiceRegistry.LLM_SERVICES if svc.value == config.llm_service),
            config.llm_service,
        )
        replace_question_with_answer("Language model:", llm_label)

        # Question 5c: TTS Service
        tts_choices = [
            Choice(
                title=svc.label,
                value=svc.value,
            )
            for svc in ServiceRegistry.TTS_SERVICES
        ]
        config.tts_service = questionary.select(
            "Text-to-Speech:",
            choices=tts_choices,
            style=custom_style,
        ).ask()

        tts_label = next(
            (svc.label for svc in ServiceRegistry.TTS_SERVICES if svc.value == config.tts_service),
            config.tts_service,
        )
        replace_question_with_answer("Text-to-Speech:", tts_label)

    else:  # realtime mode
        # Question 5d: Realtime Service
        realtime_choices = [
            Choice(
                title=svc.label,
                value=svc.value,
            )
            for svc in ServiceRegistry.REALTIME_SERVICES
        ]
        config.realtime_service = questionary.select(
            "Realtime service:",
            choices=realtime_choices,
            style=custom_style,
        ).ask()

        realtime_label = next(
            (
                svc.label
                for svc in ServiceRegistry.REALTIME_SERVICES
                if svc.value == config.realtime_service
            ),
            config.realtime_service,
        )
        replace_question_with_answer("Realtime service:", realtime_label)

    # Question 6: Feature customization gate
    console.print("\n[bold]Default feature settings:[/bold]")
    console.print("  • Audio recording: [dim]No[/dim]")
    console.print("  • Transcription logging: [dim]No[/dim]")
    if config.mode == "cascade":
        console.print("  • Smart turn-taking: [green]Yes[/green] [dim](recommended)[/dim]")
    if config.bot_type == "web":
        console.print("  • Video avatar service: [dim]No[/dim]")
        console.print("  • Video input: [dim]No[/dim]")
        console.print("  • Video output: [dim]No[/dim]")
    console.print("  • Observability: [dim]No[/dim]")

    customize_features = questionary.confirm(
        "Customize feature settings?",
        default=False,
        style=custom_style,
    ).ask()
    replace_question_with_answer(
        "Customize feature settings?", "Yes" if customize_features else "No"
    )

    if customize_features:
        # Question 6a: Recording
        config.recording = questionary.confirm(
            "Audio recording?",
            default=False,
            style=custom_style,
        ).ask()
        replace_question_with_answer("Audio recording?", "Yes" if config.recording else "No")

        # Question 6b: Transcription
        config.transcription = questionary.confirm(
            "Transcription logging?",
            default=False,
            style=custom_style,
        ).ask()
        replace_question_with_answer(
            "Transcription logging?", "Yes" if config.transcription else "No"
        )

        # Question 6d: Video avatar service (only for web/mobile bots)
        if config.bot_type == "web":
            use_video_service = questionary.confirm(
                "Use video avatar service?",
                default=False,
                style=custom_style,
            ).ask()
            replace_question_with_answer(
                "Use video avatar service?", "Yes" if use_video_service else "No"
            )

            if use_video_service:
                video_choices = [
                    Choice(
                        title=svc.label,
                        value=svc.value,
                    )
                    for svc in ServiceRegistry.VIDEO_SERVICES
                ]
                config.video_service = questionary.select(
                    "Video avatar service:",
                    choices=video_choices,
                    style=custom_style,
                ).ask()

                if config.video_service:
                    video_label = next(
                        (
                            svc.label
                            for svc in ServiceRegistry.VIDEO_SERVICES
                            if svc.value == config.video_service
                        ),
                        config.video_service,
                    )
                    replace_question_with_answer("Video avatar service:", video_label)

        # Question 6e: Video input (only for web/mobile bots)
        if config.bot_type == "web":
            config.video_input = questionary.confirm(
                "Video input?",
                default=False,
                style=custom_style,
            ).ask()
            replace_question_with_answer("Video input?", "Yes" if config.video_input else "No")
        else:
            # Telephony bots don't support video
            config.video_input = False

        # Question 6f: Video output (only for web/mobile bots, skip if video service selected)
        if config.bot_type == "web":
            if config.video_service:
                # Video service requires video output, enable automatically
                config.video_output = True
            else:
                config.video_output = questionary.confirm(
                    "Video output?",
                    default=False,
                    style=custom_style,
                ).ask()
                replace_question_with_answer(
                    "Video output?", "Yes" if config.video_output else "No"
                )
        else:
            # Telephony bots don't support video
            config.video_output = False

        # Question 6g: Observability
        config.enable_observability = questionary.confirm(
            "Enable observability?",
            default=False,
            style=custom_style,
        ).ask()
        replace_question_with_answer(
            "Enable observability?", "Yes" if config.enable_observability else "No"
        )
    else:
        # Apply default feature settings
        config.video_service = None
        config.video_input = False
        config.video_output = False
        config.recording = False
        config.transcription = False
        config.enable_observability = False

    # Question 7: Pipecat Cloud deployment
    config.deploy_to_cloud = questionary.confirm(
        "Deploy to Pipecat Cloud?",
        default=True,
        style=custom_style,
    ).ask()
    replace_question_with_answer(
        "Deploy to Pipecat Cloud?", "Yes" if config.deploy_to_cloud else "No"
    )

    # Question 8: Krisp noise cancellation (only if deploying to cloud)
    if config.deploy_to_cloud:
        config.enable_krisp = questionary.confirm(
            "Enable Krisp noise cancellation?",
            default=False,
            style=custom_style,
        ).ask()
        replace_question_with_answer(
            "Enable Krisp noise cancellation?", "Yes" if config.enable_krisp else "No"
        )

    # Question 9: Eval transport (behavioral evals). Bot-type agnostic; off by
    # default — it adds an inert "eval" entry to transport_params that is only used
    # when the bot is run with `-t eval`.
    config.enable_eval = questionary.confirm(
        "Enable evals?",
        default=False,
        style=custom_style,
    ).ask()
    replace_question_with_answer("Enable evals?", "Yes" if config.enable_eval else "No")

    return config
