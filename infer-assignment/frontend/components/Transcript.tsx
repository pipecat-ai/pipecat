"use client";

import { useEffect, useRef } from "react";
import { TranscriptEntry } from "@/lib/types";

interface TranscriptProps {
  transcripts: TranscriptEntry[];
  currentTime: number;
}

export default function Transcript({ transcripts, currentTime }: TranscriptProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (activeRef.current && containerRef.current) {
      const container = containerRef.current;
      const activeElement = activeRef.current;
      const containerRect = container.getBoundingClientRect();
      const activeRect = activeElement.getBoundingClientRect();
      
      if (activeRect.top < containerRect.top || activeRect.bottom > containerRect.bottom) {
        activeElement.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }, [currentTime]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const isActive = (entry: TranscriptEntry): boolean => {
    const endTime = entry.end_time ?? entry.start_time + 10;
    return currentTime >= entry.start_time && currentTime <= endTime;
  };

  if (transcripts.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-3 text-gray-200">Transcript</h2>
        <div className="text-gray-500 text-center py-8">
          No transcript available
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-3 text-gray-200">Transcript</h2>
      <div
        ref={containerRef}
        className="space-y-3 max-h-[400px] overflow-y-auto pr-2"
      >
        {transcripts.map((entry, index) => {
          const active = isActive(entry);
          return (
            <div
              key={index}
              ref={active ? activeRef : null}
              className={`p-3 rounded-lg transition-colors ${
                active
                  ? "bg-indigo-900/30 border border-indigo-500/50"
                  : "bg-gray-800/50"
              }`}
            >
              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className={`text-sm font-medium ${
                      entry.role === "user" ? "text-green-400" : "text-blue-400"
                    }`}
                  >
                    {entry.role === "user" ? "User" : "Assistant"}
                  </span>
                  <span className="text-xs text-gray-500 font-mono">
                    {formatTime(entry.start_time)}
                    {entry.end_time && ` - ${formatTime(entry.end_time)}`}
                  </span>
                </div>
                <p className="text-gray-300 text-sm leading-relaxed">
                  {entry.text}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
