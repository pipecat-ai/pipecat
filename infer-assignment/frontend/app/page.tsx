"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import AudioPlayer from "@/components/AudioPlayer";
import Transcript from "@/components/Transcript";
import VoiceClient from "@/components/VoiceClient";
import { fetchSession, getAudioUrl, triggerFreeze, checkFreezeStatus } from "@/lib/api";
import { SessionData } from "@/lib/types";

export default function Home() {
  const [session, setSession] = useState<SessionData | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isFreezing, setIsFreezing] = useState(false);
  const [freezeDuration, setFreezeDuration] = useState(3);
  const [hasAudio, setHasAudio] = useState(false);
  const [isVoiceConnected, setIsVoiceConnected] = useState(false);
  const initialLoadDone = useRef(false);

  const loadSession = useCallback(async (silent = false) => {
    try {
      if (!silent) setLoading(true);
      const data = await fetchSession();
      setSession(data);
      setError(null);
      
      if (data && data.duration_seconds > 0) {
        try {
          const audioResponse = await fetch(getAudioUrl(), { method: 'HEAD' });
          setHasAudio(audioResponse.ok);
        } catch {
          setHasAudio(false);
        }
      } else {
        setHasAudio(false);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to load session";
      if (!errorMessage.includes("404") && !errorMessage.includes("No session")) {
        setError(errorMessage);
      }
      setSession(null);
      setHasAudio(false);
    } finally {
      setLoading(false);
    }
  }, []);

  // Check freeze status periodically
  useEffect(() => {
    if (!isVoiceConnected) return;

    const checkFreeze = async () => {
      try {
        const status = await checkFreezeStatus();
        setIsFreezing(status.is_frozen);
      } catch {
        // todo: handle errors
      }
    };

    const interval = setInterval(checkFreeze, 1000);
    return () => clearInterval(interval);
  }, [isVoiceConnected]);

  useEffect(() => {
    initialLoadDone.current = true;
  }, []);

  const handleTriggerFreeze = async () => {
    try {
      await triggerFreeze(freezeDuration);
      setIsFreezing(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to trigger freeze");
    }
  };

  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-xl font-bold">Infer Assignment</h1>
              {session && (
                <p className="text-sm text-gray-400">
                  Session: {session.session_id}
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-gray-800 rounded-lg px-3 py-2">
              <label className="text-sm text-gray-400">Freeze:</label>
              <input
                type="number"
                min={1}
                max={10}
                value={freezeDuration}
                onChange={(e) => setFreezeDuration(Number(e.target.value))}
                className="w-14 bg-gray-700 rounded px-2 py-1 text-sm text-center"
              />
              <span className="text-sm text-gray-400">sec</span>
              <button
                onClick={handleTriggerFreeze}
                disabled={isFreezing || !session}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  isFreezing
                    ? "bg-red-600 text-white animate-pulse"
                    : "bg-blue-600 hover:bg-blue-500 text-white"
                }`}
              >
                {isFreezing ? "Freezing..." : "Trigger Freeze"}
              </button>
            </div>

          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        {error && (
          <div className="bg-red-900/30 border border-red-800 rounded-lg p-4 mb-6">
            <p className="text-red-400">{error}</p>
            <p className="text-sm text-gray-400 mt-1">
              Make sure the backend server is running at http://localhost:8000
            </p>
          </div>
        )}

        <div className="mb-6">
          <VoiceClient
            isFreezing={isFreezing}
            onSessionStart={() => {
              setIsVoiceConnected(true);
              setTimeout(() => loadSession(true), 1000);
            }}
            onSessionEnd={() => {
              setIsVoiceConnected(false);
              // Reload session to get final recording
              setTimeout(loadSession, 500);
            }}
          />
        </div>

        {loading && !session ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <p className="text-gray-400">Loading session data...</p>
            </div>
          </div>
        ) : session ? (
          <div className="max-w-4xl mx-auto space-y-6">
            {/* Audio player with waveform, latency overlay, and freeze indicators */}
            <AudioPlayer
              audioUrl={hasAudio ? getAudioUrl() : null}
              latencies={session.latencies}
              freezeEvents={session.freeze_events}
              duration={session.duration_seconds}
              onTimeUpdate={setCurrentTime}
            />

            {/* Transcript */}
            <Transcript
              transcripts={session.transcripts}
              currentTime={currentTime}
            />
          </div>
        ) : (
          <div className="text-center py-20">
            <h2 className="text-xl font-semibold text-gray-400 mb-2">
              No Session Available
            </h2>
            <p className="text-gray-500 max-w-md mx-auto mb-4">
              Click the "Connect" button to start a voice session.
              After talking with the AI, click "Disconnect" to see the recording.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
