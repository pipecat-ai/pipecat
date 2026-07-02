#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SERVICE METADATA REGISTRY - SOURCE OF TRUTH.

⭐ THIS IS THE SOURCE OF TRUTH FOR ALL PIPECAT SERVICES ⭐

To add a new service:
  1. Add a ServiceDefinition to the appropriate list below
  2. Run: uv run scripts/cli/update_registry.py  (regenerates _imports.py and _configs.py)

DO NOT edit _configs.py or _imports.py directly - they are auto-generated from this file.

This module contains:
  - ServiceDefinition dataclass
  - Service lists (WEBRTC_TRANSPORTS, STT_SERVICES, LLM_SERVICES, etc.)
  - FEATURE_DEFINITIONS dict
  - MANUAL_SERVICE_CONFIGS for complex initialization

For operations on services, use ServiceLoader from service_loader.py.
"""

from dataclasses import dataclass
from typing import Literal

DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way."

from ._configs import SERVICE_CONFIGS
from ._imports import BASE_IMPORTS, FEATURE_IMPORTS, IMPORTS

# Type aliases for service categorization
ServiceType = Literal["transport", "stt", "llm", "tts", "realtime"]
BotType = Literal["web", "telephony"]


@dataclass
class ServiceDefinition:
    """Service metadata definition.

    Required fields:
        value: Service identifier (e.g., "openai_llm")
        label: Human-readable name (e.g., "OpenAI")
        package: Python package requirement (e.g., "pipecat-ai[openai]")

    Optional fields:
        class_name: List of class names to import for this service
        env_prefix: Prefix for environment variables (e.g., "OPENAI" -> "OPENAI_API_KEY")
        include_params: Constructor params that have defaults but should still appear in the
            generated config (e.g., "api_key" has a default but we want users to set it via
            env var). Controls *whether* a param is generated.
        settings_params: Params to wrap in a Service.Settings(...) block instead of passing
            as direct constructor args. Controls *where* a param goes. When set (even if
            empty), the generator uses the Settings pattern and adds system_instruction.
            Params listed here are implicitly included (no need to also list in include_params).
        manual_config: If True, config must be manually written (not auto-generated)
        recommended: If True, this service is marked as recommended in prompts
        additional_imports: List of full import statements that can't be auto-discovered
        param_defaults: Default values for settings params. When set, the config generator
            produces os.getenv("ENV_VAR", "default") instead of os.getenv("ENV_VAR").
            Use this for params where the quickstart should work without the user
            setting the env var (e.g., model or voice defaults).
    """

    value: str
    label: str
    package: str
    class_name: list[str] | None = None
    env_prefix: str | None = None
    include_params: list[str] | None = None
    settings_params: list[str] | None = None
    manual_config: bool = False
    recommended: bool = False
    additional_imports: list[str] | None = None
    param_defaults: dict[str, str] | None = None

    def __post_init__(self):
        """Validate service definition after initialization."""
        if not self.value:
            raise ValueError("Service must have a value")
        if not self.label:
            raise ValueError("Service must have a label")
        if not self.package:
            raise ValueError("Service must have a package")


# Feature definitions with metadata for auto-generation
# Maps feature names to the list of classes/functions that need to be imported
FEATURE_DEFINITIONS: dict[str, list[str]] = {
    "recording": [
        "AudioBufferProcessor",
        "datetime",
        "io",
        "wave",
        "aiofiles",
    ],
    "transcription": ["AssistantTurnStoppedMessage", "UserTurnStoppedMessage"],
    "vad": ["SileroVADAnalyzer"],
    "pipeline": ["Pipeline", "WorkerRunner", "PipelineParams", "PipelineWorker"],
    "context": ["LLMContext", "LLMContextAggregatorPair", "LLMUserAggregatorParams"],
    "runner": [
        "load_dotenv",
        "RunnerArguments",
        "BaseTransport",
    ],
    # Queued on connect to kick off the conversation. Dial-out bots wait for the
    # callee to answer/speak first, so they don't import or use it.
    "llm_run_frame": ["LLMRunFrame"],
    "observability": ["WhiskerObserver"],
    # Imported on the standard (non-PSTN/SIP) transport path: the collapsed bot()
    # calls create_transport. Dial-out and SIP construct their transports by hand.
    "create_transport": ["create_transport"],
    # The "eval" transport entry (pc create --eval) needs EvalTransportParams so the
    # generated bot is runnable with `-t eval` for behavioral evals.
    "eval": ["EvalTransportParams"],
}


class ServiceRegistry:
    """Central registry for all Pipecat services and their configurations.

    This class contains only DATA - service definitions, import mappings,
    and feature configurations. All logic for querying and working with
    services has been moved to ServiceLoader for better separation of concerns.

    For operations like:
    - Finding services: use ServiceLoader.get_service_by_value()
    - Getting imports: use ServiceLoader.get_imports_for_services()
    - Extracting extras: use ServiceLoader.extract_extras_for_services()
    """

    # Service configs from separate module for better maintainability
    SERVICE_CONFIGS = SERVICE_CONFIGS

    # Auto-generated imports from separate module (DO NOT EDIT - regenerate with update_imports.py)
    IMPORTS = IMPORTS
    FEATURE_IMPORTS = FEATURE_IMPORTS
    BASE_IMPORTS = BASE_IMPORTS

    # Feature definitions (defined at module level for consistency)
    FEATURE_DEFINITIONS = FEATURE_DEFINITIONS

    # Web/Mobile Transports (WebRTC)
    WEBRTC_TRANSPORTS: list[ServiceDefinition] = [
        ServiceDefinition(
            value="daily",
            label="Daily (WebRTC)",
            package="pipecat-ai[daily]",
            # Bots build transports via create_transport(); only the params are needed.
            class_name=["DailyParams"],
        ),
        ServiceDefinition(
            value="smallwebrtc",
            label="SmallWebRTC",
            package="pipecat-ai[webrtc]",
            class_name=["TransportParams"],
        ),
        ServiceDefinition(
            value="websocket",
            label="WebSocket",
            package="pipecat-ai[websocket]",
            class_name=["FastAPIWebsocketParams", "ProtobufFrameSerializer"],
        ),
    ]

    # Telephony Transports
    TELEPHONY_TRANSPORTS: list[ServiceDefinition] = [
        ServiceDefinition(
            value="twilio",
            label="Twilio",
            package="pipecat-ai[websocket]",
            # create_transport sets the serializer; only params are needed. aiohttp and
            # BaseModel are kept for the get_call_info() / CallInfo personalization helper.
            class_name=["FastAPIWebsocketParams"],
            additional_imports=["import aiohttp", "from pydantic import BaseModel"],
        ),
        ServiceDefinition(
            value="twilio_daily_sip_dialin",
            label="Twilio + Daily SIP (Dial-in)",
            package="pipecat-ai[daily]",
            class_name=["DailyParams", "DailyTransport"],
            additional_imports=[
                "from server_utils import AgentRequest",
                "from twilio.rest import Client",
            ],
        ),
        ServiceDefinition(
            value="twilio_daily_sip_dialout",
            label="Twilio + Daily SIP (Dial-out)",
            package="pipecat-ai[daily]",
            class_name=["DailyParams", "DailyTransport"],
            additional_imports=[
                "from server_utils import AgentRequest, DialoutSettings",
                "from typing import Any",
            ],
        ),
        ServiceDefinition(
            value="daily_pstn_dialin",
            label="Daily PSTN (Dial-in)",
            package="pipecat-ai[daily]",
            # Dial-in uses the unified create_transport path: it arrives as a typed
            # DailyRunnerArguments and create_transport applies the dial-in settings
            # from the request body. DailyDialinRequest is used by the optional dial-in
            # personalization block.
            class_name=["DailyParams"],
            additional_imports=["from pipecat.runner.types import DailyDialinRequest"],
        ),
        ServiceDefinition(
            value="daily_pstn_dialout",
            label="Daily PSTN (Dial-out)",
            package="pipecat-ai[daily]",
            class_name=["DailyParams", "DailyTransport"],
            additional_imports=[
                "from server_utils import AgentRequest, DialoutSettings",
                "from typing import Any",
            ],
        ),
        ServiceDefinition(
            value="telnyx",
            label="Telnyx",
            package="pipecat-ai[websocket]",
            class_name=["FastAPIWebsocketParams"],
        ),
        ServiceDefinition(
            value="plivo",
            label="Plivo",
            package="pipecat-ai[websocket]",
            class_name=["FastAPIWebsocketParams"],
        ),
        ServiceDefinition(
            value="exotel",
            label="Exotel",
            package="pipecat-ai[websocket]",
            class_name=["FastAPIWebsocketParams"],
        ),
    ]

    # Speech-to-Text Services
    STT_SERVICES: list[ServiceDefinition] = [
        ServiceDefinition(
            value="assemblyai_stt",
            label="AssemblyAI",
            package="pipecat-ai[assemblyai]",
            class_name=["AssemblyAISTTService"],
            env_prefix="ASSEMBLYAI",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="aws_transcribe_stt",
            label="AWS Transcribe",
            package="pipecat-ai[aws]",
            class_name=["AWSTranscribeSTTService"],
            env_prefix="AWS",
            include_params=["aws_access_key_id", "aws_session_token", "region"],
        ),
        ServiceDefinition(
            value="azure_stt",
            label="Azure Speech",
            package="pipecat-ai[azure]",
            class_name=["AzureSTTService"],
            env_prefix="AZURE_SPEECH",
            include_params=["api_key", "region"],
        ),
        ServiceDefinition(
            value="cartesia_stt",
            label="Cartesia",
            package="pipecat-ai[cartesia]",
            class_name=["CartesiaSTTService"],
            env_prefix="CARTESIA",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="cartesia_turns_stt",
            label="Cartesia Turns",
            package="pipecat-ai[cartesia]",
            class_name=["CartesiaTurnsSTTService"],
            env_prefix="CARTESIA",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="deepgram_stt",
            label="Deepgram",
            package="pipecat-ai[deepgram]",
            class_name=["DeepgramSTTService"],
            env_prefix="DEEPGRAM",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="deepgram_flux_stt",
            label="Deepgram Flux",
            package="pipecat-ai[deepgram]",
            class_name=["DeepgramFluxSTTService"],
            env_prefix="DEEPGRAM",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="deepgram_flux_sagemaker_stt",
            label="Deepgram Flux SageMaker",
            package="pipecat-ai[deepgram,sagemaker]",
            class_name=["DeepgramFluxSageMakerSTTService"],
            env_prefix="DEEPGRAM_FLUX_SAGEMAKER_STT",
            include_params=["endpoint_name", "region"],
        ),
        ServiceDefinition(
            value="deepgram_sagemaker_stt",
            label="Deepgram SageMaker",
            package="pipecat-ai[deepgram,sagemaker]",
            class_name=["DeepgramSageMakerSTTService"],
            env_prefix="DEEPGRAM_SAGEMAKER_STT",
            include_params=["endpoint_name", "region"],
        ),
        ServiceDefinition(
            value="elevenlabs_stt",
            label="ElevenLabs",
            package="pipecat-ai[elevenlabs]",
            class_name=["ElevenLabsSTTService"],
            env_prefix="ELEVENLABS",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="elevenlabs_realtime_stt",
            label="ElevenLabs Realtime",
            package="pipecat-ai[elevenlabs]",
            class_name=["ElevenLabsRealtimeSTTService"],
            env_prefix="ELEVENLABS",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="fal_stt",
            label="Fal (Wizper)",
            package="pipecat-ai[fal]",
            class_name=["FalSTTService"],
            env_prefix="FAL",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="gladia_stt",
            label="Gladia",
            package="pipecat-ai[gladia]",
            class_name=["GladiaSTTService"],
            env_prefix="GLADIA",
            include_params=["api_key", "region"],
        ),
        ServiceDefinition(
            value="google_stt",
            label="Google Speech-to-Text",
            package="pipecat-ai[google]",
            class_name=["GoogleSTTService"],
            env_prefix="GOOGLE",
            include_params=["credentials", "location"],
        ),
        ServiceDefinition(
            value="gradium_stt",
            label="Gradium",
            package="pipecat-ai[gradium]",
            class_name=["GradiumSTTService"],
            env_prefix="GRADIUM",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="groq_stt",
            label="Groq (Whisper)",
            package="pipecat-ai[groq]",
            class_name=["GroqSTTService"],
            env_prefix="GROQ",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="mistral_stt",
            label="Mistral",
            package="pipecat-ai[mistral]",
            class_name=["MistralSTTService"],
            env_prefix="MISTRAL",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="nvidia_stt",
            label="NVIDIA",
            package="pipecat-ai[nvidia]",
            class_name=["NvidiaSTTService"],
            env_prefix="NVIDIA",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="nvidia_sagemaker_stt",
            label="NVIDIA SageMaker",
            package="pipecat-ai[aws,sagemaker]",
            class_name=["NvidiaSageMakerSTTService"],
            env_prefix="NVIDIA_SAGEMAKER_STT",
            include_params=["endpoint_name", "region"],
        ),
        ServiceDefinition(
            value="openai_stt",
            label="OpenAI (Whisper)",
            package="pipecat-ai[openai]",
            class_name=["OpenAISTTService"],
            env_prefix="OPENAI",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="openai_realtime_stt",
            label="OpenAI Realtime",
            package="pipecat-ai[openai]",
            class_name=["OpenAIRealtimeSTTService"],
            env_prefix="OPENAI",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="sarvam_stt",
            label="Sarvam",
            package="pipecat-ai[sarvam]",
            class_name=["SarvamSTTService"],
            env_prefix="SARVAM",
            include_params=["api_key"],
            settings_params=["model"],
        ),
        ServiceDefinition(
            value="soniox_stt",
            label="Soniox",
            package="pipecat-ai[soniox]",
            class_name=["SonioxSTTService"],
            env_prefix="SONIOX",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="speechmatics_stt",
            label="Speechmatics",
            package="pipecat-ai[speechmatics]",
            class_name=["SpeechmaticsSTTService"],
            env_prefix="SPEECHMATICS",
            include_params=["api_key"],
        ),
        ServiceDefinition(
            value="moonshine_stt",
            label="Moonshine",
            package="pipecat-ai[moonshine]",
            class_name=["MoonshineSTTService"],
            settings_params=["model"],
        ),
        ServiceDefinition(
            value="whisper_stt",
            label="Whisper (Local)",
            package="pipecat-ai[whisper]",
            class_name=["WhisperSTTService"],
            env_prefix="OPENAI",
            settings_params=["model"],
        ),
        ServiceDefinition(
            value="xai_stt",
            label="XAI",
            package="pipecat-ai[xai]",
            class_name=["XAISTTService"],
            env_prefix="XAI",
            include_params=["api_key"],
        ),
    ]

    # Large Language Model Services
    LLM_SERVICES: list[ServiceDefinition] = [
        ServiceDefinition(
            value="anthropic_llm",
            label="Anthropic Claude",
            package="pipecat-ai[anthropic]",
            class_name=["AnthropicLLMService"],
            env_prefix="ANTHROPIC",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="aws_bedrock_llm",
            label="AWS Bedrock",
            package="pipecat-ai[aws]",
            class_name=["AWSBedrockLLMService"],
            env_prefix="AWS",
            include_params=["aws_region"],
            settings_params=["model", "system_instruction"],
            manual_config=True,
        ),
        ServiceDefinition(
            value="azure_llm",
            label="Azure OpenAI",
            package="pipecat-ai[azure]",
            class_name=["AzureLLMService"],
            env_prefix="AZURE_CHATGPT",
            include_params=["api_key", "endpoint"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="cerebras_llm",
            label="Cerebras",
            package="pipecat-ai[cerebras]",
            class_name=["CerebrasLLMService"],
            env_prefix="CEREBRAS",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="deepseek_llm",
            label="DeepSeek",
            package="pipecat-ai[deepseek]",
            class_name=["DeepSeekLLMService"],
            env_prefix="DEEPSEEK",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="fireworks_llm",
            label="Fireworks AI",
            package="pipecat-ai[fireworks]",
            class_name=["FireworksLLMService"],
            env_prefix="FIREWORKS",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="google_gemini_llm",
            label="Google Gemini",
            package="pipecat-ai[google]",
            class_name=["GoogleLLMService"],
            env_prefix="GOOGLE",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="google_vertex_llm",
            label="Google Vertex AI",
            package="pipecat-ai[google]",
            class_name=["GoogleVertexLLMService"],
            env_prefix="GOOGLE",
            include_params=["credentials", "location", "project_id"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="xai_llm",
            label="Grok",
            package="pipecat-ai[xai]",
            class_name=["GrokLLMService"],
            env_prefix="XAI",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="groq_llm",
            label="Groq",
            package="pipecat-ai[groq]",
            class_name=["GroqLLMService"],
            env_prefix="GROQ",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="inception_llm",
            label="Inception",
            package="pipecat-ai[inception]",
            class_name=["InceptionLLMService"],
            env_prefix="INCEPTION",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="mistral_llm",
            label="Mistral",
            package="pipecat-ai[mistral]",
            class_name=["MistralLLMService"],
            env_prefix="MISTRAL",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="novita_llm",
            label="Novita",
            package="pipecat-ai[novita]",
            class_name=["NovitaLLMService"],
            env_prefix="NOVITA",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="nvidia_llm",
            label="NVIDIA",
            package="pipecat-ai[nvidia]",
            class_name=["NvidiaLLMService"],
            env_prefix="NVIDIA",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="ollama_llm",
            label="Ollama",
            package="pipecat-ai",
            class_name=["OLLamaLLMService"],
            env_prefix="OLLAMA",
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="openai_llm",
            label="OpenAI",
            package="pipecat-ai[openai]",
            class_name=["OpenAILLMService"],
            env_prefix="OPENAI",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
            param_defaults={"model": "gpt-4.1"},
        ),
        ServiceDefinition(
            value="openai_responses_llm",
            label="OpenAI Responses",
            package="pipecat-ai[openai]",
            class_name=["OpenAIResponsesLLMService"],
            env_prefix="OPENAI",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
            param_defaults={"model": "gpt-4.1"},
        ),
        ServiceDefinition(
            value="openrouter_llm",
            label="OpenRouter",
            package="pipecat-ai[openrouter]",
            class_name=["OpenRouterLLMService"],
            env_prefix="OPENROUTER",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="perplexity_llm",
            label="Perplexity",
            package="pipecat-ai[perplexity]",
            class_name=["PerplexityLLMService"],
            env_prefix="PERPLEXITY",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="qwen_llm",
            label="Qwen",
            package="pipecat-ai[qwen]",
            class_name=["QwenLLMService"],
            env_prefix="QWEN",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="sambanova_llm",
            label="SambaNova",
            package="pipecat-ai[sambanova]",
            class_name=["SambaNovaLLMService"],
            env_prefix="SAMBANOVA",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="sarvam_llm",
            label="Sarvam",
            package="pipecat-ai[sarvam]",
            class_name=["SarvamLLMService"],
            env_prefix="SARVAM",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
        ServiceDefinition(
            value="together_llm",
            label="Together AI",
            package="pipecat-ai[together]",
            class_name=["TogetherLLMService"],
            env_prefix="TOGETHER",
            include_params=["api_key"],
            settings_params=["model", "system_instruction"],
        ),
    ]

    # Text-to-Speech Services
    TTS_SERVICES: list[ServiceDefinition] = [
        ServiceDefinition(
            value="asyncai_tts",
            label="Async",
            package="pipecat-ai[asyncai]",
            class_name=["AsyncAITTSService"],
            env_prefix="ASYNCAI",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="aws_polly_tts",
            label="AWS Polly",
            package="pipecat-ai[aws]",
            class_name=["AWSPollyTTSService"],
            env_prefix="AWS",
            include_params=["region"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="azure_tts",
            label="Azure TTS",
            package="pipecat-ai[azure]",
            class_name=["AzureTTSService"],
            env_prefix="AZURE_SPEECH",
            include_params=["api_key", "region"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="camb_tts",
            label="Camb",
            package="pipecat-ai[camb]",
            class_name=["CambTTSService"],
            env_prefix="CAMB",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="cartesia_tts",
            label="Cartesia",
            package="pipecat-ai[cartesia]",
            class_name=["CartesiaTTSService"],
            env_prefix="CARTESIA",
            include_params=["api_key"],
            settings_params=["voice"],
            param_defaults={"voice": "71a7ad14-091c-4e8e-a314-022ece01c121"},
        ),
        ServiceDefinition(
            value="deepgram_tts",
            label="Deepgram",
            package="pipecat-ai[deepgram]",
            class_name=["DeepgramTTSService"],
            env_prefix="DEEPGRAM",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="deepgram_sagemaker_tts",
            label="Deepgram SageMaker",
            package="pipecat-ai[deepgram,sagemaker]",
            class_name=["DeepgramSageMakerTTSService"],
            env_prefix="DEEPGRAM_SAGEMAKER_TTS",
            include_params=["endpoint_name", "region"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="elevenlabs_tts",
            label="ElevenLabs",
            package="pipecat-ai[elevenlabs]",
            class_name=["ElevenLabsTTSService"],
            env_prefix="ELEVENLABS",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="fish_tts",
            label="Fish",
            package="pipecat-ai[fish]",
            class_name=["FishAudioTTSService"],
            env_prefix="FISH",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="gemini_tts",
            label="Gemini TTS",
            package="pipecat-ai[google]",
            class_name=["GeminiTTSService"],
            env_prefix="GOOGLE",
            include_params=["credentials"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="google_tts",
            label="Google TTS",
            package="pipecat-ai[google]",
            class_name=["GoogleTTSService"],
            env_prefix="GOOGLE",
            include_params=["credentials"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="gradium_tts",
            label="Gradium TTS",
            package="pipecat-ai[gradium]",
            class_name=["GradiumTTSService"],
            env_prefix="GRADIUM",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="groq_tts",
            label="Groq TTS",
            package="pipecat-ai[groq]",
            class_name=["GroqTTSService"],
            env_prefix="GROQ",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="hume_tts",
            label="Hume TTS",
            package="pipecat-ai[hume]",
            class_name=["HumeTTSService"],
            env_prefix="HUME",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="inworld_tts",
            label="Inworld",
            package="pipecat-ai",
            class_name=["InworldTTSService"],
            env_prefix="INWORLD",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="kokoro_tts",
            label="Kokoro",
            package="pipecat-ai[kokoro]",
            class_name=["KokoroTTSService"],
            env_prefix="KOKORO",
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="lmnt_tts",
            label="LMNT",
            package="pipecat-ai[lmnt]",
            class_name=["LmntTTSService"],
            env_prefix="LMNT",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="minimax_tts",
            label="MiniMax",
            package="pipecat-ai",
            class_name=["MiniMaxHttpTTSService"],
            env_prefix="MINIMAX",
            include_params=["api_key", "group_id"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="mistral_tts",
            label="Mistral",
            package="pipecat-ai[mistral]",
            class_name=["MistralTTSService"],
            env_prefix="MISTRAL",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="neuphonic_tts",
            label="Neuphonic",
            package="pipecat-ai[neuphonic]",
            class_name=["NeuphonicTTSService"],
            env_prefix="NEUPHONIC",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="nvidia_tts",
            label="NVIDIA",
            package="pipecat-ai[nvidia]",
            class_name=["NvidiaTTSService"],
            env_prefix="NVIDIA",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="nvidia_sagemaker_tts",
            label="NVIDIA SageMaker",
            package="pipecat-ai[aws,sagemaker]",
            class_name=["NvidiaSageMakerTTSService"],
            env_prefix="NVIDIA_SAGEMAKER_TTS",
            include_params=["endpoint_name", "region"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="openai_tts",
            label="OpenAI TTS",
            package="pipecat-ai[openai]",
            class_name=["OpenAITTSService"],
            env_prefix="OPENAI",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="piper_tts",
            label="Piper",
            package="pipecat-ai[piper]",
            class_name=["PiperTTSService"],
            env_prefix="PIPER",
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="resemble_tts",
            label="Resemble",
            package="pipecat-ai[resembleai]",
            class_name=["ResembleAITTSService"],
            env_prefix="RESEMBLE",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="rime_tts",
            label="Rime",
            package="pipecat-ai[rime]",
            class_name=["RimeTTSService"],
            env_prefix="RIME",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="sarvam_tts",
            label="Sarvam",
            package="pipecat-ai",
            class_name=["SarvamTTSService"],
            env_prefix="SARVAM",
            include_params=["api_key"],
            settings_params=["model", "voice"],
        ),
        ServiceDefinition(
            value="smallest_tts",
            label="Smallest",
            package="pipecat-ai[smallest]",
            class_name=["SmallestTTSService"],
            env_prefix="SMALLEST",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="soniox_tts",
            label="Soniox",
            package="pipecat-ai[soniox]",
            class_name=["SonioxTTSService"],
            env_prefix="SONIOX",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="xai_tts",
            label="XAI",
            package="pipecat-ai[xai]",
            class_name=["XAITTSService"],
            env_prefix="XAI",
            include_params=["api_key"],
            settings_params=["voice"],
        ),
        ServiceDefinition(
            value="xtts_tts",
            label="XTTS (Coqui)",
            package="pipecat-ai[xtts]",
            class_name=["XTTSService"],
            env_prefix="XTTS",
            include_params=["base_url"],
            settings_params=["voice"],
        ),
    ]

    # Realtime (Speech-to-Speech) Services
    REALTIME_SERVICES: list[ServiceDefinition] = [
        ServiceDefinition(
            value="aws_nova_realtime",
            label="AWS Nova Sonic",
            package="pipecat-ai[aws-nova-sonic]",
            class_name=["AWSNovaSonicLLMService"],
            env_prefix="AWS",
            include_params=[],
            manual_config=True,
        ),
        ServiceDefinition(
            value="azure_realtime",
            label="Azure Realtime",
            package="pipecat-ai[azure]",
            class_name=["AzureRealtimeLLMService"],
            env_prefix="AZURE",
            include_params=[],
            manual_config=True,
            additional_imports=[
                "from pipecat.services.openai.realtime.events import SessionProperties, InputAudioTranscription"
            ],
        ),
        ServiceDefinition(
            value="gemini_live_realtime",
            label="Gemini Live",
            package="pipecat-ai[google]",
            class_name=["GeminiLiveLLMService"],
            env_prefix="GOOGLE",
            include_params=[],
            manual_config=True,
        ),
        ServiceDefinition(
            value="gemini_vertex_live_realtime",
            label="Gemini Vertex Live",
            package="pipecat-ai[google]",
            class_name=["GeminiLiveVertexLLMService"],
            env_prefix="GOOGLE_VERTEX",
            include_params=[],
            manual_config=True,
        ),
        ServiceDefinition(
            value="xai_realtime",
            label="Grok Realtime",
            package="pipecat-ai[xai]",
            class_name=["GrokRealtimeLLMService", "SessionProperties"],
            env_prefix="XAI",
            include_params=[],
            manual_config=True,
        ),
        ServiceDefinition(
            value="openai_realtime",
            label="OpenAI Realtime",
            package="pipecat-ai[openai]",
            class_name=[
                "OpenAIRealtimeLLMService",
                "SessionProperties",
                "AudioConfiguration",
                "AudioInput",
                "InputAudioTranscription",
                "SemanticTurnDetection",
                "InputAudioNoiseReduction",
            ],
            env_prefix="OPENAI",
            include_params=[],
            manual_config=True,
        ),
        ServiceDefinition(
            value="ultravox",
            label="Ultravox",
            package="pipecat-ai[ultravox]",
            class_name=["UltravoxRealtimeLLMService", "OneShotInputParams"],
            env_prefix="ULTRAVOX",
            include_params=["api_key"],
            manual_config=True,
            additional_imports=[
                "import datetime",
            ],
        ),
    ]

    # Video Services (Avatars)
    VIDEO_SERVICES: list[ServiceDefinition] = [
        ServiceDefinition(
            value="heygen_video",
            label="HeyGen",
            package="pipecat-ai[heygen]",
            class_name=["HeyGenVideoService"],
            env_prefix="HEYGEN",
            include_params=["api_key"],
            manual_config=True,
            additional_imports=[
                "from pipecat.services.heygen.api_liveavatar import LiveAvatarNewSessionRequest",
                "from pipecat.services.heygen.client import ServiceType",
            ],
        ),
        ServiceDefinition(
            value="tavus_video",
            label="Tavus",
            package="pipecat-ai[tavus]",
            class_name=["TavusVideoService"],
            env_prefix="TAVUS",
            include_params=["api_key", "replica_id"],
        ),
        ServiceDefinition(
            value="simli_video",
            label="Simli",
            package="pipecat-ai[simli]",
            class_name=["SimliVideoService"],
            env_prefix="SIMLI",
            include_params=["api_key", "face_id"],
        ),
    ]


# Manual service configurations for services that require custom initialization
# These services have complex initialization logic that cannot be auto-generated
# (e.g., nested InputParams, SessionProperties, or other special requirements)
MANUAL_SERVICE_CONFIGS = {
    "nvidia_sagemaker_stt": (
        "NvidiaSageMakerSTTService(\n"
        '    endpoint_name=os.getenv("NVIDIA_SAGEMAKER_STT_ENDPOINT_NAME"),\n'
        '    region=os.getenv("AWS_REGION")\n'
        ")"
    ),
    "nvidia_sagemaker_tts": (
        "NvidiaSageMakerTTSService(\n"
        '    endpoint_name=os.getenv("NVIDIA_SAGEMAKER_TTS_ENDPOINT_NAME"),\n'
        '    region=os.getenv("AWS_REGION"),\n'
        "    settings=NvidiaSageMakerTTSService.Settings(\n"
        '        voice=os.getenv("NVIDIA_SAGEMAKER_TTS_VOICE_ID"),\n'
        "    ),\n"
        ")"
    ),
    "aws_bedrock_llm": (
        "AWSBedrockLLMService(\n"
        '    aws_region=os.getenv("AWS_REGION"),\n'
        "    settings=AWSBedrockLLMService.Settings(\n"
        '        model=os.getenv("AWS_BEDROCK_MODEL"),\n'
        f'        system_instruction="{DEFAULT_SYSTEM_INSTRUCTION}",\n'
        "    ),\n"
        ")"
    ),
    "azure_realtime": (
        "session_properties = SessionProperties(\n"
        '    input_audio_transcription=InputAudioTranscription(model="whisper-1"),\n'
        ")\n"
        "\n"
        "llm = AzureRealtimeLLMService(\n"
        '    api_key=os.getenv("AZURE_REALTIME_API_KEY"),\n'
        '    base_url=os.getenv("AZURE_REALTIME_BASE_URL"),\n'
        "    settings=AzureRealtimeLLMService.Settings(\n"
        "        session_properties=session_properties,\n"
        f'        system_instruction="{DEFAULT_SYSTEM_INSTRUCTION}",\n'
        "    ),\n"
        ")"
    ),
    "openai_realtime": (
        "session_properties = SessionProperties(\n"
        "    audio=AudioConfiguration(\n"
        "        input=AudioInput(\n"
        "            transcription=InputAudioTranscription(),\n"
        "            turn_detection=SemanticTurnDetection(),\n"
        '            noise_reduction=InputAudioNoiseReduction(type="near_field"),\n'
        "        )\n"
        "    ),\n"
        ")\n"
        "\n"
        "llm = OpenAIRealtimeLLMService(\n"
        '    api_key=os.getenv("OPENAI_API_KEY"),\n'
        "    settings=OpenAIRealtimeLLMService.Settings(\n"
        "        session_properties=session_properties,\n"
        f'        system_instruction="{DEFAULT_SYSTEM_INSTRUCTION}",\n'
        "    ),\n"
        ")"
    ),
    "gemini_live_realtime": (
        "llm = GeminiLiveLLMService(\n"
        '    api_key=os.getenv("GOOGLE_API_KEY"),\n'
        "    settings=GeminiLiveLLMService.Settings(\n"
        '        model=os.getenv("GOOGLE_MODEL"),\n'
        '        voice=os.getenv("GOOGLE_VOICE_ID"),\n'
        f'        system_instruction="{DEFAULT_SYSTEM_INSTRUCTION}",\n'
        "    ),\n"
        ")"
    ),
    "gemini_vertex_live_realtime": (
        "llm = GeminiLiveVertexLLMService(\n"
        '        credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),\n'
        '        project_id=os.getenv("GOOGLE_PROJECT_ID"),\n'
        '        location=os.getenv("GOOGLE_LOCATION"),\n'
        "        settings=GeminiLiveVertexLLMService.Settings(\n"
        '            model=os.getenv("GOOGLE_MODEL"),\n'
        '            voice=os.getenv("GOOGLE_VOICE_ID"),\n'
        f'            system_instruction="{DEFAULT_SYSTEM_INSTRUCTION}",\n'
        "        ),\n"
        "    ),\n"
        ")"
    ),
    "xai_realtime": (
        "session_properties = SessionProperties(\n"
        '    voice=os.getenv("XAI_VOICE_ID"),\n'
        ")\n"
        "\n"
        "llm = GrokRealtimeLLMService(\n"
        '    api_key=os.getenv("XAI_API_KEY"),\n'
        "    settings=GrokRealtimeLLMService.Settings(\n"
        "        session_properties=session_properties,\n"
        f'        system_instruction="{DEFAULT_SYSTEM_INSTRUCTION}",\n'
        "    ),\n"
        ")"
    ),
    "aws_nova_realtime": (
        "llm = AWSNovaSonicLLMService(\n"
        '    secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),\n'
        '    access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),\n'
        '    region=os.getenv("AWS_REGION"),\n'
        '    session_token=os.getenv("AWS_SESSION_TOKEN"),\n'
        "    settings=AWSNovaSonicLLMService.Settings(\n"
        '        voice=os.getenv("AWS_VOICE_ID"),\n'
        f'        system_instruction="{DEFAULT_SYSTEM_INSTRUCTION}",\n'
        "    ),\n"
        ")"
    ),
    "heygen_video": (
        "HeyGenVideoService(\n"
        '    api_key=os.getenv("HEYGEN_API_KEY"),\n'
        "    service_type=ServiceType.LIVE_AVATAR,\n"
        "    session=session,\n"
        "    session_request=LiveAvatarNewSessionRequest(\n"
        '        avatar_id="HEYGEN_AVATAR_ID",\n'
        "    ),\n"
        ")"
    ),
    "ultravox": (
        "llm =UltravoxRealtimeLLMService(\n"
        "    params=OneShotInputParams(\n"
        '        api_key=os.getenv("ULTRAVOX_API_KEY"),\n'
        '        system_prompt=os.getenv("ULTRAVOX_SYSTEM_PROMPT"),\n'
        "        temperature=0.3,\n"
        "        max_duration=datetime.timedelta(minutes=3),\n"
        "    ),\n"
        ")"
    ),
}
