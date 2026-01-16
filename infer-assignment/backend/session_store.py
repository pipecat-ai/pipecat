"""Session storage for voice agent recordings.

Stores audio recordings, transcripts, latency metrics, and freeze events
for a single session.
"""

import io
import os
import time
import wave
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class TranscriptEntry:
    """A single transcript entry with timing information."""
    role: str  # "user" or "assistant"
    text: str
    start_time: float  # Relative to session start
    end_time: Optional[float] = None
    
    def to_dict(self):
        return {
            "role": self.role,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


@dataclass
class LatencyEntry:
    """Turn latency measurement."""
    turn_index: int
    user_stop_time: float  # When user stopped speaking
    bot_start_time: float  # When bot started speaking
    latency_ms: float  # Difference in milliseconds
    
    def to_dict(self):
        return {
            "turn_index": self.turn_index,
            "user_stop_time": self.user_stop_time,
            "bot_start_time": self.bot_start_time,
            "latency_ms": self.latency_ms
        }


@dataclass
class FreezeEvent:
    """A freeze event during the session."""
    start_time: float  # Relative to session start
    end_time: float
    duration_ms: float
    
    def to_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms
        }


@dataclass
class SessionData:
    """Complete session data."""
    session_id: str
    created_at: str
    sample_rate: int = 16000
    transcripts: List[TranscriptEntry] = field(default_factory=list)
    latencies: List[LatencyEntry] = field(default_factory=list)
    freeze_events: List[FreezeEvent] = field(default_factory=list)
    audio_data: bytes = b""
    session_start_time: float = 0
    
    def to_dict(self):
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "sample_rate": self.sample_rate,
            "duration_seconds": len(self.audio_data) / (self.sample_rate * 2) if self.audio_data else 0,
            "transcripts": [t.to_dict() for t in self.transcripts],
            "latencies": [l.to_dict() for l in self.latencies],
            "freeze_events": [f.to_dict() for f in self.freeze_events]
        }


class SessionStore:
    """Manages session data storage."""
    
    def __init__(self, storage_dir: str = "recordings"):
        self.storage_dir = storage_dir
        self._session: Optional[SessionData] = None
        os.makedirs(storage_dir, exist_ok=True)
    
    def create_session(self, session_id: str, sample_rate: int = 16000) -> SessionData:
        """Create a new session."""
        self._session = SessionData(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            sample_rate=sample_rate,
            session_start_time=time.time()
        )
        return self._session
    
    def get_session(self) -> Optional[SessionData]:
        """Get the current session."""
        return self._session
    
    def add_transcript(self, role: str, text: str, start_time: Optional[float] = None):
        """Add a transcript entry."""
        if not self._session:
            return
        
        # Convert absolute time to relative time from session start
        if start_time:
            relative_start = start_time - self._session.session_start_time
        else:
            relative_start = time.time() - self._session.session_start_time
        
        entry = TranscriptEntry(
            role=role,
            text=text,
            start_time=relative_start
        )
        self._session.transcripts.append(entry)
    
    def update_transcript_end_time(self, end_time: Optional[float] = None):
        """Update the end time of the last transcript entry."""
        if not self._session or not self._session.transcripts:
            return
        
        # Convert absolute time to relative time from session start
        if end_time:
            relative_end = end_time - self._session.session_start_time
        else:
            relative_end = time.time() - self._session.session_start_time
        
        self._session.transcripts[-1].end_time = relative_end
    
    def add_latency(self, user_stop_time: float, bot_start_time: float):
        """Add a latency measurement."""
        if not self._session:
            return
        
        # Convert to relative times
        relative_user_stop = user_stop_time - self._session.session_start_time
        relative_bot_start = bot_start_time - self._session.session_start_time
        latency_ms = (bot_start_time - user_stop_time) * 1000
        
        entry = LatencyEntry(
            turn_index=len(self._session.latencies),
            user_stop_time=relative_user_stop,
            bot_start_time=relative_bot_start,
            latency_ms=latency_ms
        )
        self._session.latencies.append(entry)
    
    def add_freeze_event(self, start_time: float, end_time: float):
        """Add a freeze event."""
        if not self._session:
            return
        
        relative_start = start_time - self._session.session_start_time
        relative_end = end_time - self._session.session_start_time
        duration_ms = (end_time - start_time) * 1000
        
        event = FreezeEvent(
            start_time=relative_start,
            end_time=relative_end,
            duration_ms=duration_ms
        )
        self._session.freeze_events.append(event)
    
    def append_audio(self, audio_bytes: bytes):
        """Append audio data to the session."""
        if not self._session:
            return
        self._session.audio_data += audio_bytes
    
    def save_audio_to_file(self) -> Optional[str]:
        """Save the audio data to a WAV file."""
        if not self._session or not self._session.audio_data:
            return None
        
        filename = os.path.join(
            self.storage_dir,
            f"session_{self._session.session_id}.wav"
        )
        
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)  # 16-bit audio
                wf.setnchannels(1)  # Mono
                wf.setframerate(self._session.sample_rate)
                wf.writeframes(self._session.audio_data)
            
            with open(filename, "wb") as f:
                f.write(buffer.getvalue())
        
        return filename
    
    def get_audio_wav_bytes(self) -> Optional[bytes]:
        """Get the audio data as WAV bytes."""
        if not self._session or not self._session.audio_data:
            return None
        
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setsampwidth(2)
            wf.setnchannels(1)
            wf.setframerate(self._session.sample_rate)
            wf.writeframes(self._session.audio_data)
        
        return buffer.getvalue()


# Global session store instance
session_store = SessionStore()
