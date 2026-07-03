#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Validation and configuration building for non-interactive project creation."""

import json
from pathlib import Path

from pipecat.cli.prompts.questions import ProjectConfig
from pipecat.cli.registry import ServiceRegistry


class ConfigValidationError(Exception):
    """Raised when project configuration validation fails.

    Collects all validation errors so agents see every problem at once.
    """

    def __init__(self, errors: list[str]):
        """Initialize a ConfigValidationError.

        Args:
            errors: List of validation errors
        """
        self.errors = errors
        msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


def _get_valid_values(service_list) -> list[str]:
    """Extract valid value strings from a list of ServiceDefinitions."""
    return [svc.value for svc in service_list]


def validate_and_build_config(
    *,
    name: str | None = None,
    bot_type: str | None = None,
    transport: list[str] | None = None,
    mode: str | None = None,
    stt: str | None = None,
    llm: str | None = None,
    tts: str | None = None,
    realtime: str | None = None,
    video: str | None = None,
    client_framework: str | None = None,
    client_server: str | None = None,
    daily_pstn_mode: str | None = None,
    twilio_daily_sip_mode: str | None = None,
    recording: bool = False,
    transcription: bool = False,
    video_input: bool = False,
    video_output: bool = False,
    deploy_to_cloud: bool = True,
    enable_krisp: bool = False,
    observability: bool = False,
    enable_eval: bool = False,
) -> ProjectConfig:
    """Validate all inputs and build a ProjectConfig.

    Collects all errors before raising, so callers see every problem at once.

    Returns:
        A fully-populated ProjectConfig.

    Raises:
        ConfigValidationError: If any validation fails.
    """
    errors: list[str] = []

    # --- Required fields ---
    if not name:
        errors.append("--name is required")

    # --bot-type is optional in non-interactive mode: when omitted it is inferred
    # from the transports below (a forking question for the interactive wizard, but
    # the transports already determine it headless). Only validate it when given.
    if bot_type and bot_type not in ("web", "telephony"):
        errors.append(f"--bot-type must be 'web' or 'telephony', got '{bot_type}'")

    if not transport:
        errors.append("At least one --transport is required")

    if not mode:
        errors.append("--mode is required (cascade or realtime)")
    elif mode not in ("cascade", "realtime"):
        errors.append(f"--mode must be 'cascade' or 'realtime', got '{mode}'")

    # --- Transport validation ---
    all_transports = ServiceRegistry.WEBRTC_TRANSPORTS + ServiceRegistry.TELEPHONY_TRANSPORTS
    valid_transport_values = _get_valid_values(all_transports)
    # Also accept the generic names that need mode resolution
    generic_transport_names = {"daily_pstn", "twilio_daily_sip"}
    webrtc_values = _get_valid_values(ServiceRegistry.WEBRTC_TRANSPORTS)
    telephony_values = _get_valid_values(ServiceRegistry.TELEPHONY_TRANSPORTS)

    resolved_transports: list[str] = []
    if transport:
        for t in transport:
            if t == "daily_pstn":
                # Needs mode resolution
                if not daily_pstn_mode:
                    errors.append(
                        "--daily-pstn-mode is required when transport is 'daily_pstn' "
                        "(dial-in or dial-out)"
                    )
                elif daily_pstn_mode not in ("dial-in", "dial-out"):
                    errors.append(
                        f"--daily-pstn-mode must be 'dial-in' or 'dial-out', "
                        f"got '{daily_pstn_mode}'"
                    )
                else:
                    resolved = f"daily_pstn_{daily_pstn_mode.replace('-', '')}"
                    resolved_transports.append(resolved)
            elif t == "twilio_daily_sip":
                # Needs mode resolution
                if not twilio_daily_sip_mode:
                    errors.append(
                        "--twilio-daily-sip-mode is required when transport is 'twilio_daily_sip' "
                        "(dial-in or dial-out)"
                    )
                elif twilio_daily_sip_mode not in ("dial-in", "dial-out"):
                    errors.append(
                        f"--twilio-daily-sip-mode must be 'dial-in' or 'dial-out', "
                        f"got '{twilio_daily_sip_mode}'"
                    )
                else:
                    resolved = f"twilio_daily_sip_{twilio_daily_sip_mode.replace('-', '')}"
                    resolved_transports.append(resolved)
            elif t in valid_transport_values:
                resolved_transports.append(t)
            else:
                all_valid = sorted(set(valid_transport_values) | generic_transport_names)
                errors.append(f"Unknown transport '{t}'. Valid transports: {', '.join(all_valid)}")

    # Infer bot_type from the transports when it wasn't given: telephony if any
    # transport is a telephony transport, otherwise web. (Explicit values are
    # validated against the transports by the cross-check below.)
    if not bot_type and resolved_transports:
        bot_type = "telephony" if any(t in telephony_values for t in resolved_transports) else "web"

    # Cross-check transport vs bot_type
    if bot_type and resolved_transports:
        for t in resolved_transports:
            if bot_type == "web" and t in telephony_values:
                errors.append(f"Transport '{t}' is a telephony transport but bot-type is 'web'")
            elif bot_type == "telephony" and t in webrtc_values:
                # WebRTC transports are allowed for telephony bots (local testing)
                pass

    # --- Mode-specific service validation ---
    if mode == "cascade":
        if not stt:
            errors.append("--stt is required for cascade mode")
        elif stt not in _get_valid_values(ServiceRegistry.STT_SERVICES):
            errors.append(
                f"Unknown STT service '{stt}'. "
                f"Valid: {', '.join(_get_valid_values(ServiceRegistry.STT_SERVICES))}"
            )

        if not llm:
            errors.append("--llm is required for cascade mode")
        elif llm not in _get_valid_values(ServiceRegistry.LLM_SERVICES):
            errors.append(
                f"Unknown LLM service '{llm}'. "
                f"Valid: {', '.join(_get_valid_values(ServiceRegistry.LLM_SERVICES))}"
            )

        if not tts:
            errors.append("--tts is required for cascade mode")
        elif tts not in _get_valid_values(ServiceRegistry.TTS_SERVICES):
            errors.append(
                f"Unknown TTS service '{tts}'. "
                f"Valid: {', '.join(_get_valid_values(ServiceRegistry.TTS_SERVICES))}"
            )

        if realtime:
            errors.append("--realtime should not be specified in cascade mode")

    elif mode == "realtime":
        if not realtime:
            errors.append("--realtime is required for realtime mode")
        elif realtime not in _get_valid_values(ServiceRegistry.REALTIME_SERVICES):
            errors.append(
                f"Unknown realtime service '{realtime}'. "
                f"Valid: {', '.join(_get_valid_values(ServiceRegistry.REALTIME_SERVICES))}"
            )

        if stt:
            errors.append("--stt should not be specified in realtime mode")
        if llm:
            errors.append("--llm should not be specified in realtime mode")
        if tts:
            errors.append("--tts should not be specified in realtime mode")

    # --- Video service validation ---
    if video:
        if video not in _get_valid_values(ServiceRegistry.VIDEO_SERVICES):
            errors.append(
                f"Unknown video service '{video}'. "
                f"Valid: {', '.join(_get_valid_values(ServiceRegistry.VIDEO_SERVICES))}"
            )
        if bot_type == "telephony":
            errors.append("Video services are only available for web bots")

    # --- Client validation ---
    generate_client = False
    resolved_client_framework = client_framework
    resolved_client_server = client_server

    if client_framework:
        if client_framework not in ("react", "vanilla", "none"):
            errors.append(
                f"--client-framework must be 'react', 'vanilla', or 'none', "
                f"got '{client_framework}'"
            )
        if bot_type == "telephony":
            errors.append("--client-framework is only available for web bots")
        if client_framework == "react":
            generate_client = True
            if client_server and client_server not in ("vite", "nextjs"):
                errors.append(f"--client-server must be 'vite' or 'nextjs', got '{client_server}'")
        elif client_framework == "vanilla":
            generate_client = True
            resolved_client_server = "vite"
        elif client_framework == "none":
            resolved_client_framework = None

    if client_server and not client_framework:
        errors.append("--client-server requires --client-framework")

    # --- Cross-field constraints ---
    if video_input and bot_type == "telephony":
        errors.append("--video-input is only available for web bots")
    if video_output and bot_type == "telephony":
        errors.append("--video-output is only available for web bots")
    if enable_krisp and not deploy_to_cloud:
        errors.append("--enable-krisp requires --deploy-to-cloud")

    # --- Daily PSTN mode without matching transport ---
    if daily_pstn_mode and transport and "daily_pstn" not in transport:
        has_pstn = any(t.startswith("daily_pstn") for t in transport)
        if not has_pstn:
            errors.append("--daily-pstn-mode specified but no 'daily_pstn' transport")
    if twilio_daily_sip_mode and transport and "twilio_daily_sip" not in transport:
        has_sip = any(t.startswith("twilio_daily_sip") for t in transport)
        if not has_sip:
            errors.append("--twilio-daily-sip-mode specified but no 'twilio_daily_sip' transport")

    # Bail out with all errors
    if errors:
        raise ConfigValidationError(errors)

    # --- Apply defaults ---

    # Force video output on if a video service is selected
    if video:
        video_output = True

    # Resolve daily_pstn_mode / twilio_daily_sip_mode from the resolved transport names
    resolved_daily_pstn_mode = None
    resolved_twilio_daily_sip_mode = None
    for t in resolved_transports:
        if t == "daily_pstn_dialin":
            resolved_daily_pstn_mode = "dial-in"
        elif t == "daily_pstn_dialout":
            resolved_daily_pstn_mode = "dial-out"
        elif t == "twilio_daily_sip_dialin":
            resolved_twilio_daily_sip_mode = "dial-in"
        elif t == "twilio_daily_sip_dialout":
            resolved_twilio_daily_sip_mode = "dial-out"

    # Validation above raises ConfigValidationError on any collected error before
    # reaching here, so these required fields are present and valid; assert to narrow
    # the Optional CLI parameters for the type checker.
    assert name is not None
    assert bot_type in ("web", "telephony")
    assert mode is not None

    config = ProjectConfig(
        project_name=name,
        bot_type=bot_type,
        transports=resolved_transports,
        mode=mode,
        stt_service=stt if mode == "cascade" else None,
        llm_service=llm if mode == "cascade" else None,
        tts_service=tts if mode == "cascade" else None,
        realtime_service=realtime if mode == "realtime" else None,
        video_service=video,
        generate_client=generate_client,
        client_framework=resolved_client_framework,
        client_server=resolved_client_server,
        daily_pstn_mode=resolved_daily_pstn_mode,
        twilio_daily_sip_mode=resolved_twilio_daily_sip_mode,
        video_input=video_input,
        video_output=video_output,
        recording=recording,
        transcription=transcription,
        deploy_to_cloud=deploy_to_cloud,
        enable_krisp=enable_krisp,
        enable_observability=observability,
        enable_eval=enable_eval,
    )
    return config


def load_config_from_file(path: Path) -> dict:
    """Load a JSON config file and return it as a dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path) as f:
        return json.load(f)


def config_to_json(config: ProjectConfig) -> str:
    """Serialize a ProjectConfig to a pretty-printed JSON string."""
    data = {
        "project_name": config.project_name,
        "bot_type": config.bot_type,
        "transports": config.transports,
        "mode": config.mode,
        "stt_service": config.stt_service,
        "llm_service": config.llm_service,
        "tts_service": config.tts_service,
        "realtime_service": config.realtime_service,
        "video_service": config.video_service,
        "generate_client": config.generate_client,
        "client_framework": config.client_framework,
        "client_server": config.client_server,
        "daily_pstn_mode": config.daily_pstn_mode,
        "twilio_daily_sip_mode": config.twilio_daily_sip_mode,
        "video_input": config.video_input,
        "video_output": config.video_output,
        "recording": config.recording,
        "transcription": config.transcription,
        "deploy_to_cloud": config.deploy_to_cloud,
        "enable_krisp": config.enable_krisp,
        "enable_observability": config.enable_observability,
        "enable_eval": config.enable_eval,
    }
    return json.dumps(data, indent=2)
