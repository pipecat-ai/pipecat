"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import WaveSurfer from "wavesurfer.js";
import Timeline from "wavesurfer.js/dist/plugins/timeline.esm.js";
import { FreezeEvent, LatencyEntry } from "@/lib/types";

interface AudioPlayerProps {
  audioUrl: string | null;
  latencies: LatencyEntry[];
  freezeEvents: FreezeEvent[];
  duration: number;
  onTimeUpdate?: (time: number) => void;
}

export default function AudioPlayer({
  audioUrl,
  latencies,
  freezeEvents,
  duration,
  onTimeUpdate,
}: AudioPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    // Don't initialize if no audio URL or no container
    if (!containerRef.current || !audioUrl) {
      setIsLoading(false);
      return;
    }

    let wavesurfer: WaveSurfer | null = null;
    let isMounted = true;

    const initWaveSurfer = async () => {
      try {
        const response = await fetch(audioUrl, { method: 'HEAD' });
        if (!response.ok) {
          if (isMounted) {
            setError("No audio recording available yet");
            setIsLoading(false);
          }
          return;
        }

        if (!isMounted || !containerRef.current || !timelineRef.current) return;

        wavesurfer = WaveSurfer.create({
          container: containerRef.current,
          waveColor: "#4f46e5",
          progressColor: "#818cf8",
          cursorColor: "#f59e0b",
          cursorWidth: 2,
          barWidth: 2,
          barGap: 1,
          barRadius: 2,
          height: 100,
          normalize: true,
          backend: "WebAudio",
          plugins: [
            Timeline.create({
              container: timelineRef.current,
              height: 20,
              primaryLabelInterval: 5,
              secondaryLabelInterval: 1,
              secondaryLabelOpacity: 0.5,
              style: {
                color: "#9ca3af",
                fontSize: "11px",
                fontFamily: "monospace",
              },
              formatTimeCallback: (seconds: number) => {
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return `${mins}:${secs.toString().padStart(2, "0")}`;
              },
            }),
          ],
        });

        wavesurfer.on("ready", () => {
          if (isMounted) {
            setIsLoading(false);
            setIsReady(true);
            setError(null);
          }
        });

        wavesurfer.on("error", (err) => {
          if (isMounted) {
            setError(err.message || "Failed to load audio");
            setIsLoading(false);
          }
        });

        wavesurfer.on("play", () => {
          if (isMounted) setIsPlaying(true);
        });
        wavesurfer.on("pause", () => {
          if (isMounted) setIsPlaying(false);
        });
        wavesurfer.on("finish", () => {
          if (isMounted) setIsPlaying(false);
        });

        wavesurfer.on("timeupdate", (time) => {
          if (isMounted) {
            setCurrentTime(time);
            onTimeUpdate?.(time);
          }
        });

        wavesurferRef.current = wavesurfer;

        // Load audio
        await wavesurfer.load(audioUrl);
      } catch (err) {
        if (isMounted) {
          setError("No audio recording available yet");
          setIsLoading(false);
        }
      }
    };

    setIsLoading(true);
    setError(null);
    setIsReady(false);
    initWaveSurfer();

    return () => {
      isMounted = false;
      if (wavesurfer) {
        try {
          wavesurfer.destroy();
        } catch {
          // todo: handle errors
        }
      }
      wavesurferRef.current = null;
    };
  }, [audioUrl, onTimeUpdate]);

  const togglePlayPause = useCallback(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause();
    }
  }, []);

  const restart = useCallback(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.seekTo(0);
      wavesurferRef.current.play();
    }
  }, []);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // position percentage for overlays
  const getPositionPercent = (time: number): number => {
    if (duration <= 0) return 0;
    return (time / duration) * 100;
  };

  // If no audio URL, show placeholder
  if (!audioUrl) {
    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <div className="flex items-center gap-4 mb-4">
          <button
            disabled
            className="px-4 py-2 rounded bg-gray-700 text-gray-500"
          >
            Play
          </button>
          <div className="text-sm text-gray-500">
            <span className="font-mono">0:00</span>
            <span className="mx-1">/</span>
            <span className="font-mono">0:00</span>
          </div>
        </div>
        <div className="relative min-h-[100px] flex items-center justify-center bg-gray-800/50 rounded">
          <div className="text-center text-gray-500">
            <p>No recording available</p>
            <p className="text-xs mt-1">Start a voice session to record audio</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <div className="flex items-center gap-4 mb-4">
        <button
          onClick={togglePlayPause}
          disabled={isLoading || !isReady}
          className="px-4 py-2 rounded bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 text-white transition-colors"
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button
          onClick={restart}
          disabled={isLoading || !isReady}
          className="px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 text-gray-300 transition-colors"
        >
          Restart
        </button>
        <div className="text-sm text-gray-400">
          <span className="text-white font-mono">{formatTime(currentTime)}</span>
          <span className="mx-1">/</span>
          <span className="font-mono">{formatTime(duration)}</span>
        </div>
      </div>

      <div className="relative waveform-container">
        {isReady && freezeEvents.map((freeze, index) => (
          <div
            key={`freeze-${index}`}
            className="absolute top-0 bottom-0 freeze-region z-10"
            style={{
              left: `${getPositionPercent(freeze.start_time)}%`,
              width: `${getPositionPercent(freeze.end_time - freeze.start_time)}%`,
            }}
            title={`Freeze: ${freeze.duration_ms.toFixed(0)}ms`}
          >
            <div className="absolute top-1 left-1 text-xs text-red-400 bg-red-900/80 px-1 rounded">
              🥶 {freeze.duration_ms.toFixed(0)}ms
            </div>
          </div>
        ))}

        {isReady && latencies.flatMap((latency, index) => [
          // Marker 1: User stops speaking
          <div
            key={`user-stop-${index}`}
            className="latency-marker"
            style={{ left: `${getPositionPercent(latency.user_stop_time)}%` }}
          >
            <div className="latency-label" style={{ fontSize: '10px' }}>
              U
            </div>
          </div>,
          // Marker 2: Bot starts speaking (with latency value)
          <div
            key={`bot-start-${index}`}
            className="latency-marker"
            style={{ left: `${getPositionPercent(latency.bot_start_time)}%` }}
          >
            <div className="latency-label">
              {latency.latency_ms.toFixed(0)}ms
            </div>
          </div>
        ])}

        <div ref={containerRef} className="min-h-[100px]" />

        <div ref={timelineRef} className="timeline-container" />

        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 rounded">
            <div className="text-gray-400">Loading audio...</div>
          </div>
        )}

        {error && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 rounded">
            <div className="text-center text-gray-500">
              <p className="text-sm">{error}</p>
            </div>
          </div>
        )}
      </div>

      <div className="flex gap-4 mt-3 text-xs text-gray-500">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-yellow-500 rounded-sm" />
          <span>Turn latency</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-red-500/50 border border-red-500 rounded-sm" />
          <span>Freeze events</span>
        </div>
      </div>
    </div>
  );
}
