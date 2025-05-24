"""Audio analysis module for measuring actual latency from recordings.

This module provides tools to analyze audio recordings and extract real latency
measurements by detecting silence periods and comparing them with event-based metrics.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


@dataclass
class SilenceSegment:
    """Represents a detected silence period in audio."""

    start_time: float
    end_time: float
    duration: float  # in milliseconds
    confidence: float  # 0-1, how confident we are this is actual silence


@dataclass
class AudioLatencyMeasurement:
    """Represents an actual latency measurement from audio analysis."""

    type: str  # 'response_latency' or 'interruption_latency'
    duration_ms: float
    start_time: float
    end_time: float
    confidence: float
    details: Dict


@dataclass
class LatencyComparison:
    """Comparison between event-based and audio-based metrics."""

    session_id: str
    event_based_ms: float
    audio_based_ms: float
    variance_ms: float
    variance_percent: float
    status: str  # 'accurate', 'warning', 'error'


class AudioAnalyzer:
    """Analyzes audio recordings to detect actual silence periods and latencies."""

    def __init__(
        self,
        silence_threshold: float = 0.01,  # RMS threshold for silence
        min_silence_duration: float = 0.5,  # Minimum silence duration in seconds
        sample_rate: int = 16000,
    ):
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate

    def load_session_recordings(self) -> Dict:
        """Load session recordings data from the JSON file."""
        session_recordings_path = "session_data/session_recordings.json"

        if not os.path.exists(session_recordings_path):
            raise FileNotFoundError(f"Session recordings file not found: {session_recordings_path}")

        try:
            with open(session_recordings_path, "r") as f:
                session_recordings = json.load(f)
            return session_recordings
        except Exception as e:
            raise Exception(f"Error loading session recordings: {e}")

    def get_full_recording_path(self, session_id: str) -> str:
        """Get the full recording file path for a given session ID."""
        session_recordings = self.load_session_recordings()

        if session_id not in session_recordings:
            raise ValueError(f"Session {session_id} not found in recordings data")

        session_data = session_recordings[session_id]

        # Look for full recordings first
        if "full" in session_data and session_data["full"]:
            # Get the most recent full recording
            full_recordings = session_data["full"]
            if isinstance(full_recordings, list) and len(full_recordings) > 0:
                # Use the most recent full recording
                return full_recordings[-1]
            elif isinstance(full_recordings, str):
                return full_recordings

        # If no full recording, try to find any recording file
        for recording_type in ["user", "bot"]:
            if recording_type in session_data and session_data[recording_type]:
                recordings = session_data[recording_type]
                if isinstance(recordings, list) and len(recordings) > 0:
                    print(
                        f"⚠️  No full recording found for session {session_id}, using {recording_type} recording"
                    )
                    return recordings[-1]
                elif isinstance(recordings, str):
                    print(
                        f"⚠️  No full recording found for session {session_id}, using {recording_type} recording"
                    )
                    return recordings

        raise FileNotFoundError(f"No recordings found for session {session_id}")

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio file, convert to mono if needed
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio file {audio_path}: {e}")

    def detect_silence_segments(self, audio: np.ndarray, sr: int) -> List[SilenceSegment]:
        """Detect periods of silence in the audio."""
        # Calculate RMS energy in small windows
        hop_length = 512
        frame_length = 2048

        # Calculate RMS energy for each frame
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

        # Convert frame indices to time
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        # Identify silence frames (below threshold)
        silence_frames = rms < self.silence_threshold

        # Group consecutive silence frames into segments
        silence_segments = []
        start_idx = None

        for i, is_silent in enumerate(silence_frames):
            if is_silent and start_idx is None:
                # Start of silence
                start_idx = i
            elif not is_silent and start_idx is not None:
                # End of silence
                start_time = times[start_idx]
                end_time = times[i - 1] if i > 0 else times[i]
                duration = (end_time - start_time) * 1000  # Convert to ms

                # Only keep silence segments longer than minimum duration
                if duration >= self.min_silence_duration * 1000:
                    # Calculate confidence based on how silent the segment really is
                    segment_rms = rms[start_idx:i]
                    avg_rms = np.mean(segment_rms) if len(segment_rms) > 0 else 0
                    confidence = max(0, 1 - (avg_rms / self.silence_threshold))

                    silence_segments.append(
                        SilenceSegment(
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            confidence=confidence,
                        )
                    )

                start_idx = None

        # Handle case where audio ends with silence
        if start_idx is not None:
            start_time = times[start_idx]
            end_time = times[-1]
            duration = (end_time - start_time) * 1000

            if duration >= self.min_silence_duration * 1000:
                segment_rms = rms[start_idx:]
                avg_rms = np.mean(segment_rms) if len(segment_rms) > 0 else 0
                confidence = max(0, 1 - (avg_rms / self.silence_threshold))

                silence_segments.append(
                    SilenceSegment(
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        confidence=confidence,
                    )
                )

        return silence_segments

    def extract_response_latencies(
        self, silence_segments: List[SilenceSegment]
    ) -> List[AudioLatencyMeasurement]:
        """Extract response latency measurements from silence segments."""
        response_latencies = []

        for segment in silence_segments:
            # For now, consider all significant silence periods as potential response latencies
            # In a more sophisticated version, we could use additional heuristics
            # like segment position, duration patterns, etc.

            # Only consider silences that are likely response gaps (not too short, not too long)
            if 50 <= segment.duration <= 5000:  # Between 50ms and 5 seconds
                response_latencies.append(
                    AudioLatencyMeasurement(
                        type="response_latency",
                        duration_ms=segment.duration,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        confidence=segment.confidence,
                        details={
                            "detection_method": "silence_analysis",
                            "rms_threshold": self.silence_threshold,
                            "min_duration_ms": self.min_silence_duration * 1000,
                        },
                    )
                )

        return response_latencies

    def analyze_session_audio(self, session_id: str) -> Dict:
        """Analyze audio file for a given session and extract latency measurements."""
        try:
            # Get the full recording path from session recordings data
            audio_path = self.get_full_recording_path(session_id)
            print(f"📊 Analyzing audio for session {session_id}: {audio_path}")

            # Load and analyze audio
            audio, sr = self.load_audio(audio_path)

            # Detect silence segments
            silence_segments = self.detect_silence_segments(audio, sr)

            # Extract response latency measurements
            response_latencies = self.extract_response_latencies(silence_segments)

            # Calculate summary statistics
            if response_latencies:
                durations = [m.duration_ms for m in response_latencies]
                avg_duration = np.mean(durations)
                min_duration = np.min(durations)
                max_duration = np.max(durations)
                confidence_avg = np.mean([m.confidence for m in response_latencies])
            else:
                avg_duration = min_duration = max_duration = confidence_avg = 0

            result = {
                "session_id": session_id,
                "audio_file": audio_path,
                "audio_duration_seconds": len(audio) / sr,
                "analysis_parameters": {
                    "silence_threshold": self.silence_threshold,
                    "min_silence_duration_ms": self.min_silence_duration * 1000,
                    "sample_rate": sr,
                },
                "analysis_summary": {
                    "total_silence_segments": len(silence_segments),
                    "response_latency_count": len(response_latencies),
                    "avg_response_latency_ms": round(avg_duration, 2) if response_latencies else 0,
                    "min_response_latency_ms": round(min_duration, 2) if response_latencies else 0,
                    "max_response_latency_ms": round(max_duration, 2) if response_latencies else 0,
                    "avg_confidence": round(confidence_avg, 3) if response_latencies else 0,
                },
                "silence_segments": [
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "duration_ms": seg.duration,
                        "confidence": seg.confidence,
                    }
                    for seg in silence_segments
                ],
                "response_latencies": [
                    {
                        "type": lat.type,
                        "duration_ms": lat.duration_ms,
                        "start_time": lat.start_time,
                        "end_time": lat.end_time,
                        "confidence": lat.confidence,
                        "details": lat.details,
                    }
                    for lat in response_latencies
                ],
            }

            return result

        except Exception as e:
            raise Exception(f"Error analyzing audio for session {session_id}: {e}")

    def compare_with_client_metrics(self, session_id: str) -> List[LatencyComparison]:
        """Compare audio-based measurements with client-side event metrics."""
        try:
            # Load audio analysis results
            audio_results = self.analyze_session_audio(session_id)
            # Load client metrics
            client_metrics_path = f"session_data/{session_id}_client_metrics.json"
            if not os.path.exists(client_metrics_path):
                raise FileNotFoundError(f"No client metrics found for session {session_id}")

            with open(client_metrics_path, "r") as f:
                client_metrics_data = json.load(f)

            # Extract metrics from client data
            client_metrics = client_metrics_data.get("metrics", [])

            # Get client response latencies and user latencies
            client_response_latencies = [
                m for m in client_metrics if m.get("type") == "response_latency"
            ]

            client_user_latencies = [m for m in client_metrics if m.get("type") == "user_latency"]

            # Get total silence duration from audio analysis
            silence_segments = audio_results.get("silence_segments", [])
            silence_segments = silence_segments[1:-1] if len(silence_segments) >= 2 else []
            total_silence_duration_ms = sum([seg["duration_ms"] for seg in silence_segments])

            comparisons = []

            # Compare adjusted audio response latency with client response latency
            if client_response_latencies:
                # Calculate totals from client metrics
                client_response_total = sum([m["value_ms"] for m in client_response_latencies])
                client_user_total = sum([m["value_ms"] for m in client_user_latencies])

                # Calculate adjusted audio response latency
                # Total silence - user latency = response latency
                adjusted_audio_response_latency = total_silence_duration_ms - client_user_total

                # Ensure we don't get negative values due to measurement errors
                adjusted_audio_response_latency = max(0, adjusted_audio_response_latency)

                variance_ms = abs(adjusted_audio_response_latency - client_response_total)
                variance_percent = (
                    (variance_ms / client_response_total * 100) if client_response_total > 0 else 0
                )

                # Determine status based on variance
                if variance_percent <= 10:
                    status = "accurate"
                elif variance_percent <= 25:
                    status = "warning"
                else:
                    status = "error"

                comparisons.append(
                    LatencyComparison(
                        session_id=session_id,
                        event_based_ms=client_response_total,
                        audio_based_ms=adjusted_audio_response_latency,
                        variance_ms=variance_ms,
                        variance_percent=variance_percent,
                        status=status,
                    )
                )

            return comparisons

        except Exception as e:
            raise Exception(f"Error comparing metrics for session {session_id}: {e}")


def analyze_session_audio(
    session_id: str, silence_threshold: float = 0.01, min_silence_duration: float = 0.5
) -> Dict:
    """Convenience function to analyze audio for a session."""
    analyzer = AudioAnalyzer(
        silence_threshold=silence_threshold, min_silence_duration=min_silence_duration
    )
    return analyzer.analyze_session_audio(session_id)


def compare_session_latencies(session_id: str) -> List[LatencyComparison]:
    """Convenience function to compare latencies for a session."""
    analyzer = AudioAnalyzer()
    return analyzer.compare_with_client_metrics(session_id)
