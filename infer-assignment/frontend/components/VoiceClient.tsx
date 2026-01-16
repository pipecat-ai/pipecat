"use client";

import { useState, useRef, useCallback, useEffect } from "react";

interface VoiceClientProps {
  onSessionStart?: () => void;
  onSessionEnd?: () => void;
  isFreezing?: boolean;
}

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";

export default function VoiceClient({ onSessionStart, onSessionEnd, isFreezing }: VoiceClientProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const playbackQueueRef = useRef<ArrayBuffer[]>([]);
  const playbackSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const isPlayingRef = useRef(false);
  const isMountedRef = useRef(true);
  const isFreezingRef = useRef(false);

  // When freeze is triggered, stop all audio playback immediately
  useEffect(() => {
    isFreezingRef.current = isFreezing || false;
    if (isFreezing) {
      console.log("🥶 Freeze detected - stopping audio playback");
      playbackQueueRef.current = [];
      playbackSourcesRef.current.forEach(source => {
        try {
          source.stop();
        } catch {
          // Ignore - source may have already finished
        }
      });
      playbackSourcesRef.current = [];
      isPlayingRef.current = false;
    }
  }, [isFreezing]);

  // Play received audio
  const playAudio = useCallback(async (audioData: ArrayBuffer) => {
    const audioContext = audioContextRef.current;
    if (!audioContext || audioContext.state === 'closed') return;

    if (isFreezingRef.current) {
      return;
    }

    try {
      // Convert PCM16 to Float32 for Web Audio API
      const int16Array = new Int16Array(audioData);
      const float32Array = new Float32Array(int16Array.length);
      for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768;
      }

      const audioBuffer = audioContext.createBuffer(
        1,
        float32Array.length,
        16000
      );
      audioBuffer.getChannelData(0).set(float32Array);

      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      
      playbackSourcesRef.current.push(source);
      
      source.onended = () => {
        const index = playbackSourcesRef.current.indexOf(source);
        if (index > -1) {
          playbackSourcesRef.current.splice(index, 1);
        }
      };
      
      source.start();
    } catch (err) {
      console.error("Error playing audio:", err);
    }
  }, []);

  const processPlaybackQueue = useCallback(async () => {
    if (isPlayingRef.current || playbackQueueRef.current.length === 0) return;

    isPlayingRef.current = true;
    while (playbackQueueRef.current.length > 0) {
      const audioData = playbackQueueRef.current.shift();
      if (audioData) {
        await playAudio(audioData);
        // Small delay between chunks for smoother playback
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
    }
    isPlayingRef.current = false;
  }, [playAudio]);

  const connect = useCallback(async () => {
    // Prevent double connection
    if (wsRef.current || isConnecting) return;
    
    try {
      setIsConnecting(true);
      setError(null);

      // Get microphone access first (this prompts the user)
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      });
      
      if (!isMountedRef.current) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      
      mediaStreamRef.current = stream;

      // Create audio context after getting microphone
      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      // Create WebSocket connection
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        if (!isMountedRef.current) {
          ws.close();
          return;
        }
        
        console.log("WebSocket connected");
        setIsConnected(true);
        setIsConnecting(false);
        onSessionStart?.();

        // Ensure audio context is ready
        const ctx = audioContextRef.current;
        if (!ctx || ctx.state === 'closed') {
          console.error("AudioContext not available");
          return;
        }

        // Start sending audio
        try {
          const source = ctx.createMediaStreamSource(stream);
          sourceRef.current = source;
          const processor = ctx.createScriptProcessor(4096, 1, 1);
          processorRef.current = processor;

          processor.onaudioprocess = (e) => {
            if (ws.readyState === WebSocket.OPEN) {
              const inputData = e.inputBuffer.getChannelData(0);
              const pcm16 = new Int16Array(inputData.length);
              for (let i = 0; i < inputData.length; i++) {
                pcm16[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
              }
              ws.send(pcm16.buffer);
            }
          };

          source.connect(processor);
          processor.connect(ctx.destination);
        } catch (audioErr) {
          console.error("Error setting up audio:", audioErr);
        }
      };

      ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          // Queue audio for playback
          playbackQueueRef.current.push(event.data);
          processPlaybackQueue();
        }
      };

      ws.onerror = (event) => {
        console.error("WebSocket error:", event);
        if (isMountedRef.current) {
          setError("Connection error - is the backend running?");
        }
      };

      ws.onclose = () => {
        console.log("WebSocket closed");
        if (isMountedRef.current) {
          setIsConnected(false);
          setIsConnecting(false);
        }
        onSessionEnd?.();
      };
    } catch (err) {
      if (isMountedRef.current) {
        const errorMessage = err instanceof Error ? err.message : "Failed to connect";
        // Provide helpful message for permission errors
        if (errorMessage.toLowerCase().includes("permission denied") || 
            errorMessage.toLowerCase().includes("notallowederror")) {
          setError("Microphone access denied. Please allow microphone access in your browser settings and refresh the page.");
        } else if (errorMessage.toLowerCase().includes("notfounderror")) {
          setError("No microphone found. Please connect a microphone and refresh the page.");
        } else {
          setError(errorMessage);
        }
        setIsConnecting(false);
      }
    }
  }, [onSessionStart, onSessionEnd, processPlaybackQueue, isConnecting]);

  const cleanupResources = useCallback(() => {
    // Disconnect source first
    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch {
        // Ignore
      }
      sourceRef.current = null;
    }

    // Disconnect processor
    if (processorRef.current) {
      try {
        processorRef.current.disconnect();
      } catch {
        // Ignore
      }
      processorRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        // Ignore
      }
      wsRef.current = null;
    }

    // Stop media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch {
        // Ignore
      }
      audioContextRef.current = null;
    }

    // Clear playback queue
    playbackQueueRef.current = [];
    isPlayingRef.current = false;
  }, []);

  const disconnect = useCallback(() => {
    const wasConnected = isConnected;
    cleanupResources();

    if (isMountedRef.current) {
      setIsConnected(false);
      setIsConnecting(false);
      setError(null);
    }
    
    if (wasConnected) {
      onSessionEnd?.();
    }
  }, [onSessionEnd, cleanupResources, isConnected]);

  // Set mounted state and cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    // Clear any stale error on mount
    setError(null);
    
    return () => {
      isMountedRef.current = false;
      // Only cleanup resources, don't trigger onSessionEnd
      cleanupResources();
    };
  }, [cleanupResources]);

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-4 text-gray-200">Voice Connection</h2>

      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-3 mb-4 text-sm text-red-400">
          {error}
        </div>
      )}

      <div className="flex items-center gap-3">
        {!isConnected ? (
          <button
            onClick={connect}
            disabled={isConnecting}
            className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:bg-gray-700 rounded-lg transition-colors"
          >
            {isConnecting ? "Connecting..." : "Connect"}
          </button>
        ) : (
          <button
            onClick={disconnect}
            className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded-lg transition-colors"
          >
            Disconnect
          </button>
        )}

        <div className="flex items-center gap-2 ml-auto">
          <div
            className={`w-2 h-2 rounded-full ${
              isConnected ? "bg-green-500 animate-pulse" : "bg-gray-500"
            }`}
          />
          <span className="text-sm text-gray-400">
            {isConnected ? "Connected" : isConnecting ? "Connecting..." : "Disconnected"}
          </span>
        </div>
      </div>
    </div>
  );
}
