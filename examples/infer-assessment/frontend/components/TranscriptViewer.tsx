"use client";

import { useEffect, useRef } from "react";
import { TranscriptTurn } from "../lib/api";
import { User, Bot } from "lucide-react";

interface TranscriptViewerProps {
    transcript: TranscriptTurn[];
    sessionStartTime: number;
    currentTime: number;
}

export function TranscriptViewer({ transcript, sessionStartTime, currentTime }: TranscriptViewerProps) {
    const scrollRef = useRef<HTMLDivElement>(null);

    // Map audio time back to transcript items using explicit session start time
    const startTime = sessionStartTime;

    // Find active turn
    const activeIndex = transcript.findIndex((turn, i) => {
        const turnTime = turn.timestamp - startTime;
        const nextTurnTime = i + 1 < transcript.length ? transcript[i + 1].timestamp - startTime : Infinity;
        return currentTime >= turnTime && currentTime < nextTurnTime;
    });

    useEffect(() => {
        // Scroll active item into view
        if (activeIndex !== -1 && scrollRef.current) {
            const activeEl = scrollRef.current.children[activeIndex] as HTMLElement;
            if (activeEl) {
                activeEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    }, [activeIndex]);

    return (
        <div className="flex flex-col gap-4 overflow-y-auto max-h-[500px] p-2" ref={scrollRef}>
            {transcript.map((turn, i) => {
                const isActive = i === activeIndex;
                const isUser = turn.role === "user";

                return (
                    <div
                        key={i}
                        className={`flex gap-3 p-3 rounded-lg transition-colors ${isActive ? 'bg-blue-50 ring-1 ring-blue-100' : ''}`}
                    >
                        <div className={`mt-1 flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${isUser ? 'bg-gray-200 text-gray-600' : 'bg-blue-100 text-blue-600'}`}>
                            {isUser ? <User size={16} /> : <Bot size={16} />}
                        </div>

                        <div className="flex-1">
                            <div className="flex justify-between items-baseline mb-1">
                                <span className="font-semibold text-sm capitalize text-gray-900">{turn.role}</span>
                                {turn.role === 'assistant' && turn.latency > 0 && (
                                    <span className="text-xs px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-medium">
                                        Latency: {turn.latency.toFixed(2)}s
                                    </span>
                                )}
                            </div>
                            <div className="text-gray-700 leading-relaxed text-sm">
                                {turn.content}
                            </div>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
