import ssl
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


@dataclass
class SpeechFragment:
    """Fragment of an utterance."""

    start_time: float
    """Start time of the fragment in seconds."""

    end_time: float
    """End time of the fragment in seconds."""

    language: str = "en"
    """Language of the fragment."""

    is_eos: bool = False
    """Whether the fragment is the end of a sentence."""

    is_final: bool = False
    """Whether the fragment is the final fragment."""

    attaches_to: str = ""
    """Whether the fragment attaches to the previous or next fragment (punctuation)."""

    content: str = ""
    """Content of the fragment."""

    speaker: Optional[str] = None
    """Speaker of the fragment."""

    confidence: float = 1.0
    """Confidence of the fragment."""


@dataclass
class RTConversationConfig:
    """Conversation config."""

    end_of_utterance_silence_trigger: Optional[float] = None
    """How much silence in seconds is required to trigger end of utterance detection."""


@dataclass
class RTSpeakerDiarizationConfig:
    """Real-time mode: Speaker diarization config."""

    max_speakers: Optional[int] = None
    """This enforces the maximum number of speakers allowed in a single audio stream."""

    speaker_sensitivity: Optional[float] = None
    """The sensitivity of the speaker detection.
    This is a number between 0 and 1, where 0 means least sensitive and 1 means
    most sensitive."""

    prefer_current_speaker: Optional[bool] = None
    """Whether to prefer the current speaker when assigning speaker labels.
    If true, the algorithm will prefer to stay with the current active speaker if it
    is a close enough match, even if other speakers may be closer.  This is useful
    for cases where we can flip incorrectly between similar speakers during a single
    speaker section."""

    speakers: Optional[dict[str, list[str]]] = None
    """The list of speakers detected in the audio stream."""


@dataclass
class TranscriptionConfig:
    """Real-time: Defines transcription parameters.

    See https://docs.speechmatics.com/rt-api-ref#transcription-config
    """

    language: str = "en"
    """ISO 639-1 language code. eg. `en`"""

    operating_point: Optional[str] = None
    """Specifies which acoustic model to use."""

    output_locale: Optional[str] = None
    """RFC-5646 language code for transcript output. eg. `en-AU`"""

    diarization: Optional[str] = None
    """Indicates type of diarization to use, if any."""

    additional_vocab: Optional[dict] = None
    """Additional vocabulary that is not part of the standard language."""

    punctuation_overrides: Optional[dict] = None
    """Permitted puctuation marks for advanced punctuation."""

    enable_entities: Optional[bool] = None
    """Indicates if inverse text normalization entity output is enabled."""

    max_delay: Optional[float] = None
    """Maximum acceptable delay."""

    max_delay_mode: Optional[str] = None
    """Determines whether the threshold specified in max_delay can be exceeded
    if a potential entity is detected. Flexible means if a potential entity
    is detected, then the max_delay can be overriden until the end of that
    entity. Fixed means that max_delay specified ignores any potential
    entity that would not be completed within that threshold."""

    enable_partials: Optional[bool] = None
    """Indicates if partials for transcription, where words are produced
    immediately, is enabled."""

    audio_filtering_config: Optional[dict] = None
    """Puts a lower limit on the volume of processed audio by using the volume_threshold setting."""

    transcript_filtering_config: Optional[dict] = None
    """Removes disfluencies with the remove_disfluencies setting."""

    speaker_diarization_config: Optional[RTSpeakerDiarizationConfig] = None
    """Options for speaker diarization such as ``max_speakers``."""

    conversation_config: Optional[RTConversationConfig] = None
    """Optional configuration for end-of-utterance detection."""

    def asdict(self) -> dict[Any, Any]:
        """Returns model as a dict while excluding None values recursively."""
        return asdict(self, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})


@dataclass
class AudioSettings:
    """Real-time: Defines audio parameters."""

    encoding: str = "pcm_s16le"
    """Encoding format when raw audio is used. Allowed values are
    `pcm_f32le`, `pcm_s16le` and `mulaw`."""

    sample_rate: int = 16000
    """Sampling rate in hertz."""

    chunk_size: int = 256
    """Chunk size in bytes."""

    def asdict(self) -> dict[str, Any]:
        """Returns model as a dict while excluding None values recursively."""
        return {
            "type": "raw",
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
        }


@dataclass
class ConnectionSettings:
    """Defines connection parameters."""

    url: str
    """Websocket server endpoint."""

    ssl_context: ssl.SSLContext = field(default_factory=ssl.create_default_context)
    """SSL context."""

    api_key: Optional[str] = None
    """api key to authenticate a customer."""

    get_access_token: Optional[bool] = True
    """Automatically generate a temporary token for authentication."""


class ClientMessageType(str, Enum):
    # pylint: disable=invalid-name
    """Real-time: Defines various messages sent from client to server."""

    StartRecognition = "StartRecognition"
    """Initiates a recognition job based on configuration set previously."""

    AddAudio = "AddAudio"
    """Adds more audio data to the recognition job. The server confirms
    receipt by sending an :py:attr:`ServerMessageType.AudioAdded` message."""

    EndOfStream = "EndOfStream"
    """Indicates that the client has no more audio to send."""

    SetRecognitionConfig = "SetRecognitionConfig"
    """Allows the client to re-configure the recognition session."""


class ServerMessageType(str, Enum):
    """Real-time: Defines various message types sent from server to client."""

    RecognitionStarted = "RecognitionStarted"
    """Server response to :py:attr:`ClientMessageType.StartRecognition`,
    acknowledging that a recognition session has started."""

    AudioAdded = "AudioAdded"
    """Server response to :py:attr:`ClientMessageType.AddAudio`, indicating
    that audio has been added successfully."""

    AddPartialTranscript = "AddPartialTranscript"
    """Indicates a partial transcript, which is an incomplete transcript that
    is immediately produced and may change as more context becomes available.
    """

    AddTranscript = "AddTranscript"
    """Indicates the final transcript of a part of the audio."""

    EndOfUtterance = "EndOfUtterance"
    """Indicates the end of an utterance (defined with `end_of_utterance_silence_trigger`)."""

    EndOfTranscript = "EndOfTranscript"
    """Server response to :py:attr:`ClientMessageType.EndOfStream`,
    after the server has finished sending all :py:attr:`AddTranscript`
    messages."""

    Info = "Info"
    """Indicates a generic info message."""

    Warning = "Warning"
    """Indicates a generic warning message."""

    Error = "Error"
    """Indicates n generic error message."""
