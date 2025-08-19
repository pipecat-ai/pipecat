#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Runner session argument types for the development runner.

These types are used by the development runner to pass transport-specific
information to bot functions.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import WebSocket


@dataclass
class RunnerArguments:
    """Base class for runner session arguments."""

    handle_sigint: bool = field(init=False)
    handle_sigterm: bool = field(init=False)
    pipeline_idle_timeout_secs: int = field(init=False)

    def __post_init__(self):
        self.handle_sigint = False
        self.handle_sigterm = False
        self.pipeline_idle_timeout_secs = 300


@dataclass
class DailyRunnerArguments(RunnerArguments):
    """Daily transport session arguments for the runner.

    Parameters:
        room_url: Daily room URL to join
        token: Authentication token for the room
        body: Additional request data
    """

    room_url: str
    token: Optional[str] = None
    body: Optional[Any] = field(default_factory=dict)


@dataclass
class WebSocketRunnerArguments(RunnerArguments):
    """WebSocket transport session arguments for the runner.

    Parameters:
        websocket: WebSocket connection for audio streaming
    """

    websocket: WebSocket


@dataclass
class SmallWebRTCRunnerArguments(RunnerArguments):
    """Small WebRTC transport session arguments for the runner.

    Parameters:
        webrtc_connection: Pre-configured WebRTC peer connection
    """

    webrtc_connection: Any


@dataclass
class SmallWebRTCPrebuiltArguments:
    """Arguments for Small WebRTC prebuilt console template.

    Parameters:
        no_rtvi: Disables RTVI related functionality. Default: False
        server_rtvi_version: Specifies the RTVI version in use by the server. Default: None
        no_user_audio: Disables user audio input entirely. Default: False
        no_user_video: Disables user video input entirely. Default: False
        user_video_enabled: Enables user video input. Default: False
        user_audio_enabled: Enables user audio input. Default: True
        no_audio_output: Disables audio output for the bot. Default: False
        no_bot_audio: Disables audio visualization for the bot. Default: False
        no_bot_video: Disables video visualization for the bot. Default: True
        transport_type: Type of transport to use for the RTVI client. Default: "smallwebrtc"
        transport_options: Options for configuring the transport. Default: None
        connect_params: Parameters for connecting to the transport. Default: None
        client_options: Options for configuring the RTVI client. Default: None
        theme: Theme to use for the UI. Default: "system"
        no_theme_switch: Disables the theme switcher in the header. Default: False
        no_logo: Disables the logo in the header. Default: False
        no_session_info: Disables the session info panel. Default: False
        no_status_info: Disables the status info panel. Default: False
        title_text: Title displayed in the header. Default: "Pipecat Playground"
        assistant_label_text: Label for assistant messages. Default: "assistant"
        user_label_text: Label for user messages. Default: "user"
        system_label_text: Label for system messages. Default: "system"
        collapse_info_panel: Whether to collapse the info panel by default. Default: False
        collapse_media_panel: Whether to collapse the media panel by default. Default: False
    """

    # RTVI configuration
    no_rtvi: bool = False
    server_rtvi_version: Optional[str] = None

    # Media configuration
    no_user_audio: bool = False
    no_user_video: bool = False
    user_video_enabled: bool = False
    user_audio_enabled: bool = True
    no_audio_output: bool = False
    no_bot_audio: bool = False
    no_bot_video: bool = True

    # Client & transport configuration
    transport_type: str = "smallwebrtc"
    transport_options: Optional[Any] = None
    connect_params: Optional[Any] = None
    client_options: Optional[Any] = None

    # UI configuration
    theme: str = "system"
    no_theme_switch: bool = False
    no_logo: bool = False
    no_session_info: bool = False
    no_status_info: bool = False
    title_text: str = "Pipecat Playground"
    assistant_label_text: str = "assistant"
    user_label_text: str = "user"
    system_label_text: str = "system"
    collapse_info_panel: bool = False
    collapse_media_panel: bool = False
