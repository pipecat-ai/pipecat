"""Frame Type ID system for O(1) frame type dispatch.

Replaces O(n) isinstance() checks with integer comparisons.
Each frame class is assigned a unique 16-bit type ID with an 8-bit category prefix:

    type_id = (category << 8) | sub_type

Category checks: ``(frame.type_id & 0xFF00) == FrameCategory.AUDIO``
Exact checks:    ``frame.type_id == FrameType.AUDIO_RAW_INPUT``

This module is backward-compatible: isinstance() continues to work,
and type_id is an *additional* fast-path for hot loops.
"""

from __future__ import annotations


class FrameCategory:
    """8-bit category prefixes (high byte of type_id)."""

    BASE = 0x00  # Frame, SystemFrame, DataFrame, ControlFrame
    AUDIO = 0x01
    TEXT = 0x02
    IMAGE = 0x03
    VIDEO = 0x04
    LLM = 0x05
    STT = 0x06
    TTS = 0x07
    CONTROL = 0x08
    SYSTEM = 0x09
    FUNCTION = 0x0A
    USER = 0x0B  # User speaking events
    BOT = 0x0C  # Bot speaking events
    METRIC = 0x0D
    ERROR = 0x0E
    MISC = 0x0F
    DTMF = 0x10
    TASK = 0x11
    SERVICE = 0x12
    VISION = 0x13
    TRANSPORT = 0x14
    FILTER = 0x15
    MIXER = 0x16


# Auto-incrementing type counters per category.
_type_counters: dict[int, int] = {}


def _next_type_id(category: int) -> int:
    """Generate the next unique type ID for a given category."""
    sub = _type_counters.get(category, 0) + 1
    if sub > 0xFF:
        raise OverflowError(f"Category 0x{category:02X} exceeded 255 subtypes")
    _type_counters[category] = sub
    return (category << 8) | sub


class FrameType:
    """Registry of all known frame type IDs.

    Each constant is a 16-bit int: ``(category << 8) | sub_type``.
    Call ``register(name, category)`` to add new types at runtime.
    """

    # --- Base hierarchy (category 0x00) ---
    FRAME = (FrameCategory.BASE << 8) | 0x01
    SYSTEM_FRAME = (FrameCategory.BASE << 8) | 0x02
    DATA_FRAME = (FrameCategory.BASE << 8) | 0x03
    CONTROL_FRAME = (FrameCategory.BASE << 8) | 0x04

    # --- Audio (category 0x01) ---
    AUDIO_RAW_INPUT = _next_type_id(FrameCategory.AUDIO)  # InputAudioRawFrame
    AUDIO_RAW_OUTPUT = _next_type_id(FrameCategory.AUDIO)  # OutputAudioRawFrame
    AUDIO_TTS = _next_type_id(FrameCategory.AUDIO)  # TTSAudioRawFrame
    AUDIO_SPEECH = _next_type_id(FrameCategory.AUDIO)  # SpeechOutputAudioRawFrame
    AUDIO_MIX = _next_type_id(FrameCategory.AUDIO)  # MixerAudioRawFrame
    AUDIO_SILENCE = _next_type_id(FrameCategory.AUDIO)  # SilenceAudioRawFrame
    AUDIO_USER = _next_type_id(FrameCategory.AUDIO)  # UserAudioRawFrame
    AUDIO_SPEECH_CTRL = _next_type_id(FrameCategory.AUDIO)  # SpeechControlParamsFrame

    # --- Text (category 0x02) ---
    TEXT_PLAIN = _next_type_id(FrameCategory.TEXT)  # TextFrame
    TEXT_LLM = _next_type_id(FrameCategory.TEXT)  # LLMTextFrame
    TEXT_TRANSCRIPTION = _next_type_id(FrameCategory.TEXT)  # TranscriptionFrame
    TEXT_INTERIM_TRANS = _next_type_id(FrameCategory.TEXT)  # InterimTranscriptionFrame
    TEXT_AGGREGATED = _next_type_id(FrameCategory.TEXT)  # AggregatedTextFrame
    TEXT_TTS = _next_type_id(FrameCategory.TEXT)  # TTSTextFrame
    TEXT_TRANSLATION = _next_type_id(FrameCategory.TEXT)  # TranslationFrame
    TEXT_INPUT_RAW = _next_type_id(FrameCategory.TEXT)  # InputTextRawFrame

    # --- Image (category 0x03) ---
    IMAGE_OUTPUT = _next_type_id(FrameCategory.IMAGE)  # OutputImageRawFrame
    IMAGE_URL = _next_type_id(FrameCategory.IMAGE)  # URLImageRawFrame
    IMAGE_SPRITE = _next_type_id(FrameCategory.IMAGE)  # SpriteFrame
    IMAGE_INPUT = _next_type_id(FrameCategory.IMAGE)  # InputImageRawFrame
    IMAGE_USER = _next_type_id(FrameCategory.IMAGE)  # UserImageRawFrame
    IMAGE_ASSISTANT = _next_type_id(FrameCategory.IMAGE)  # AssistantImageRawFrame

    # --- LLM (category 0x05) ---
    LLM_CONTEXT = _next_type_id(FrameCategory.LLM)
    LLM_MESSAGES = _next_type_id(FrameCategory.LLM)
    LLM_MESSAGES_APPEND = _next_type_id(FrameCategory.LLM)
    LLM_RESPONSE_START = _next_type_id(FrameCategory.LLM)
    LLM_RESPONSE_END = _next_type_id(FrameCategory.LLM)
    LLM_RUN = _next_type_id(FrameCategory.LLM)
    LLM_TOOL_CALL = _next_type_id(FrameCategory.LLM)
    LLM_TOOL_RESULT = _next_type_id(FrameCategory.LLM)
    LLM_THOUGHT_TEXT = _next_type_id(FrameCategory.LLM)  # LLMThoughtTextFrame
    LLM_THOUGHT_START = _next_type_id(FrameCategory.LLM)  # LLMThoughtStartFrame
    LLM_THOUGHT_END = _next_type_id(FrameCategory.LLM)  # LLMThoughtEndFrame
    LLM_MESSAGES_UPDATE = _next_type_id(FrameCategory.LLM)  # LLMMessagesUpdateFrame
    LLM_SET_TOOLS = _next_type_id(FrameCategory.LLM)  # LLMSetToolsFrame
    LLM_SET_TOOL_CHOICE = _next_type_id(FrameCategory.LLM)  # LLMSetToolChoiceFrame
    LLM_ENABLE_CACHING = _next_type_id(FrameCategory.LLM)  # LLMEnablePromptCachingFrame
    LLM_CONFIGURE_OUTPUT = _next_type_id(FrameCategory.LLM)  # LLMConfigureOutputFrame
    LLM_CTX_SUMMARY_REQ = _next_type_id(FrameCategory.LLM)  # LLMContextSummaryRequestFrame
    LLM_CTX_SUMMARY_RESULT = _next_type_id(FrameCategory.LLM)  # LLMContextSummaryResultFrame
    LLM_UPDATE_SETTINGS = _next_type_id(FrameCategory.LLM)  # LLMUpdateSettingsFrame
    LLM_CTX_ASST_TIMESTAMP = _next_type_id(FrameCategory.LLM)  # LLMContextAssistantTimestampFrame
    LLM_CTX_ASST_TS_OPENAI = _next_type_id(
        FrameCategory.LLM
    )  # OpenAILLMContextAssistantTimestampFrame

    # --- STT (category 0x06) ---
    STT_MUTE = _next_type_id(FrameCategory.STT)
    STT_UPDATE_SETTINGS = _next_type_id(FrameCategory.STT)
    STT_LANGUAGE_UPDATE = _next_type_id(FrameCategory.STT)
    STT_TRANSCRIPTION_UPDATE = _next_type_id(FrameCategory.STT)  # TranscriptionUpdateFrame
    STT_METADATA = _next_type_id(FrameCategory.STT)  # STTMetadataFrame

    # --- TTS (category 0x07) ---
    TTS_STARTED = _next_type_id(FrameCategory.TTS)
    TTS_STOPPED = _next_type_id(FrameCategory.TTS)
    TTS_UPDATE_SETTINGS = _next_type_id(FrameCategory.TTS)
    TTS_SPEAK = _next_type_id(FrameCategory.TTS)  # TTSSpeakFrame

    # --- Control (category 0x08) ---
    CTRL_START = _next_type_id(FrameCategory.CONTROL)
    CTRL_END = _next_type_id(FrameCategory.CONTROL)
    CTRL_STOP = _next_type_id(FrameCategory.CONTROL)
    CTRL_CANCEL = _next_type_id(FrameCategory.CONTROL)
    CTRL_INTERRUPT = _next_type_id(FrameCategory.CONTROL)
    CTRL_END_TASK = _next_type_id(FrameCategory.CONTROL)
    CTRL_START_INTERRUPT = _next_type_id(FrameCategory.CONTROL)  # StartInterruptionFrame
    CTRL_PAUSE = _next_type_id(FrameCategory.CONTROL)  # FrameProcessorPauseFrame
    CTRL_RESUME = _next_type_id(FrameCategory.CONTROL)  # FrameProcessorResumeFrame
    CTRL_PAUSE_URGENT = _next_type_id(FrameCategory.CONTROL)  # FrameProcessorPauseUrgentFrame
    CTRL_RESUME_URGENT = _next_type_id(FrameCategory.CONTROL)  # FrameProcessorResumeUrgentFrame
    CTRL_OUTPUT_READY = _next_type_id(FrameCategory.CONTROL)  # OutputTransportReadyFrame
    CTRL_VAD_UPDATE = _next_type_id(FrameCategory.CONTROL)  # VADParamsUpdateFrame

    # --- System (category 0x09) ---
    SYS_HEARTBEAT = _next_type_id(FrameCategory.SYSTEM)
    SYS_METRICS = _next_type_id(FrameCategory.SYSTEM)
    SYS_USER_IDLE_TIMEOUT_UPDATE = _next_type_id(FrameCategory.SYSTEM)  # UserIdleTimeoutUpdateFrame

    # --- Function calls (category 0x0A) ---
    FUNC_CALL_PROGRESS = _next_type_id(FrameCategory.FUNCTION)
    FUNC_CALL_RESULT = _next_type_id(FrameCategory.FUNCTION)
    FUNC_CALLS_STARTED = _next_type_id(FrameCategory.FUNCTION)  # FunctionCallsStartedFrame
    FUNC_CALL_CANCEL = _next_type_id(FrameCategory.FUNCTION)  # FunctionCallCancelFrame

    # --- User events (category 0x0B) ---
    USER_STARTED_SPEAKING = _next_type_id(FrameCategory.USER)
    USER_STOPPED_SPEAKING = _next_type_id(FrameCategory.USER)
    USER_MUTE_STARTED = _next_type_id(FrameCategory.USER)  # UserMuteStartedFrame
    USER_MUTE_STOPPED = _next_type_id(FrameCategory.USER)  # UserMuteStoppedFrame
    USER_SPEAKING = _next_type_id(FrameCategory.USER)  # UserSpeakingFrame
    USER_EMULATE_STARTED = _next_type_id(FrameCategory.USER)  # EmulateUserStartedSpeakingFrame
    USER_EMULATE_STOPPED = _next_type_id(FrameCategory.USER)  # EmulateUserStoppedSpeakingFrame
    USER_VAD_STARTED = _next_type_id(FrameCategory.USER)  # VADUserStartedSpeakingFrame
    USER_VAD_STOPPED = _next_type_id(FrameCategory.USER)  # VADUserStoppedSpeakingFrame
    USER_IMAGE_REQUEST = _next_type_id(FrameCategory.USER)  # UserImageRequestFrame

    # --- Bot events (category 0x0C) ---
    BOT_STARTED_SPEAKING = _next_type_id(FrameCategory.BOT)
    BOT_STOPPED_SPEAKING = _next_type_id(FrameCategory.BOT)
    BOT_SPEAKING = _next_type_id(FrameCategory.BOT)  # BotSpeakingFrame

    # --- Error (category 0x0E) ---
    ERROR_GENERAL = _next_type_id(FrameCategory.ERROR)
    ERROR_FATAL = _next_type_id(FrameCategory.ERROR)  # FatalErrorFrame

    # --- DTMF (category 0x10) ---
    DTMF_OUTPUT = _next_type_id(FrameCategory.DTMF)  # OutputDTMFFrame
    DTMF_INPUT = _next_type_id(FrameCategory.DTMF)  # InputDTMFFrame
    DTMF_OUTPUT_URGENT = _next_type_id(FrameCategory.DTMF)  # OutputDTMFUrgentFrame

    # --- Task (category 0x11) ---
    TASK_FRAME = _next_type_id(FrameCategory.TASK)  # TaskFrame
    TASK_CANCEL = _next_type_id(FrameCategory.TASK)  # CancelTaskFrame
    TASK_STOP = _next_type_id(FrameCategory.TASK)  # StopTaskFrame
    TASK_INTERRUPTION = _next_type_id(FrameCategory.TASK)  # InterruptionTaskFrame
    TASK_BOT_INTERRUPT = _next_type_id(FrameCategory.TASK)  # BotInterruptionFrame

    # --- Service (category 0x12) ---
    SERVICE_METADATA = _next_type_id(FrameCategory.SERVICE)  # ServiceMetadataFrame
    SERVICE_UPDATE = _next_type_id(FrameCategory.SERVICE)  # ServiceUpdateSettingsFrame
    SERVICE_SWITCHER = _next_type_id(FrameCategory.SERVICE)  # ServiceSwitcherFrame
    SERVICE_SWITCH_MANUAL = _next_type_id(FrameCategory.SERVICE)  # ManuallySwitchServiceFrame
    SERVICE_SWITCHER_META = _next_type_id(
        FrameCategory.SERVICE
    )  # ServiceSwitcherRequestMetadataFrame

    # --- Vision (category 0x13) ---
    VISION_TEXT = _next_type_id(FrameCategory.VISION)  # VisionTextFrame
    VISION_RESP_START = _next_type_id(FrameCategory.VISION)  # VisionFullResponseStartFrame
    VISION_RESP_END = _next_type_id(FrameCategory.VISION)  # VisionFullResponseEndFrame

    # --- Transport (category 0x14) ---
    TRANSPORT_MSG_OUT = _next_type_id(FrameCategory.TRANSPORT)  # OutputTransportMessageFrame
    TRANSPORT_MSG_BIDIR = _next_type_id(FrameCategory.TRANSPORT)  # TransportMessageFrame
    TRANSPORT_MSG_IN = _next_type_id(FrameCategory.TRANSPORT)  # InputTransportMessageFrame
    TRANSPORT_MSG_IN_URGENT = _next_type_id(
        FrameCategory.TRANSPORT
    )  # InputTransportMessageUrgentFrame
    TRANSPORT_MSG_OUT_URGENT = _next_type_id(
        FrameCategory.TRANSPORT
    )  # OutputTransportMessageUrgentFrame
    TRANSPORT_MSG_BIDIR_URGENT = _next_type_id(
        FrameCategory.TRANSPORT
    )  # TransportMessageUrgentFrame

    # --- Filter (category 0x15) ---
    FILTER_CONTROL = _next_type_id(FrameCategory.FILTER)  # FilterControlFrame
    FILTER_UPDATE = _next_type_id(FrameCategory.FILTER)  # FilterUpdateSettingsFrame
    FILTER_ENABLE = _next_type_id(FrameCategory.FILTER)  # FilterEnableFrame

    # --- Mixer (category 0x16) ---
    MIXER_CONTROL = _next_type_id(FrameCategory.MIXER)  # MixerControlFrame
    MIXER_UPDATE = _next_type_id(FrameCategory.MIXER)  # MixerUpdateSettingsFrame
    MIXER_ENABLE = _next_type_id(FrameCategory.MIXER)  # MixerEnableFrame

    # --- Runtime registration ---
    _registry: dict[str, int] = {}

    @classmethod
    def register(cls, name: str, category: int) -> int:
        """Register a new frame type at runtime.

        Returns the assigned type_id.
        """
        tid = _next_type_id(category)
        cls._registry[name] = tid
        setattr(cls, name, tid)
        return tid

    @classmethod
    def get(cls, name: str) -> int:
        """Look up a type ID by name."""
        return getattr(cls, name, 0)


# ── Fast category checks ────────────────────────────────────────────


def is_audio_frame(type_id: int) -> bool:
    """Check whether a type_id belongs to the AUDIO category."""
    return (type_id & 0xFF00) == (FrameCategory.AUDIO << 8)


def is_text_frame(type_id: int) -> bool:
    """Check whether a type_id belongs to the TEXT category."""
    return (type_id & 0xFF00) == (FrameCategory.TEXT << 8)


def is_image_frame(type_id: int) -> bool:
    """Check whether a type_id belongs to the IMAGE category."""
    return (type_id & 0xFF00) == (FrameCategory.IMAGE << 8)


def is_llm_frame(type_id: int) -> bool:
    """Check whether a type_id belongs to the LLM category."""
    return (type_id & 0xFF00) == (FrameCategory.LLM << 8)


def is_control_frame(type_id: int) -> bool:
    """Check whether a type_id belongs to the CONTROL category."""
    return (type_id & 0xFF00) == (FrameCategory.CONTROL << 8)


def is_system_frame(type_id: int) -> bool:
    """Check whether a type_id belongs to the SYSTEM category."""
    return (type_id & 0xFF00) == (FrameCategory.SYSTEM << 8)


def is_user_event(type_id: int) -> bool:
    """Check whether a type_id belongs to the USER category."""
    return (type_id & 0xFF00) == (FrameCategory.USER << 8)


def is_bot_event(type_id: int) -> bool:
    """Check whether a type_id belongs to the BOT category."""
    return (type_id & 0xFF00) == (FrameCategory.BOT << 8)


def is_dtmf_frame(type_id: int) -> bool:
    """Check whether a type_id belongs to the DTMF category."""
    return (type_id & 0xFF00) == (FrameCategory.DTMF << 8)


def is_task_frame(type_id: int) -> bool:
    """Check whether a type_id belongs to the TASK category."""
    return (type_id & 0xFF00) == (FrameCategory.TASK << 8)


def is_service_frame(type_id: int) -> bool:
    """Check whether a type_id belongs to the SERVICE category."""
    return (type_id & 0xFF00) == (FrameCategory.SERVICE << 8)


def is_transport_frame(type_id: int) -> bool:
    """Check whether a type_id belongs to the TRANSPORT category."""
    return (type_id & 0xFF00) == (FrameCategory.TRANSPORT << 8)


def is_filter_frame(type_id: int) -> bool:
    """Check whether a type_id belongs to the FILTER category."""
    return (type_id & 0xFF00) == (FrameCategory.FILTER << 8)


def is_mixer_frame(type_id: int) -> bool:
    """Check whether a type_id belongs to the MIXER category."""
    return (type_id & 0xFF00) == (FrameCategory.MIXER << 8)
