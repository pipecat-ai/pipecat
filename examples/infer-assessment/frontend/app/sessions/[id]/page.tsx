"use client";

import { useEffect, useState } from "react";
import { fetchSession, Session } from "../../../lib/api"; // Adjust import path
import { AudioPlayer } from "../../../components/AudioPlayer";
import { TranscriptViewer } from "../../../components/TranscriptViewer";
import Link from "next/link";
import { ArrowLeft, Calendar, BarChart } from "lucide-react";
import { useParams } from "next/navigation";

export default function SessionDetail() {
    const params = useParams();
    // useParams returns string | string[] | undefined. We expect 'id' to be a string.
    const id = Array.isArray(params.id) ? params.id[0] : params.id;

    const [session, setSession] = useState<Session | null>(null);
    const [loading, setLoading] = useState(true);
    const [currentTime, setCurrentTime] = useState(0);

    useEffect(() => {
        if (!id) return;

        fetchSession(id)
            .then((data) => {
                setSession(data);
                setLoading(false);
            })
            .catch((err) => {
                console.error(err);
                setLoading(false);
            });
    }, [id]);

    if (loading) return (
        <div className="min-h-screen flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
    );

    if (!session) return <div className="p-8 text-center text-red-500">Session not found</div>;

    return (
        <main className="min-h-screen bg-gray-50 p-8">
            <div className="max-w-4xl mx-auto">
                <Link
                    href="/"
                    className="inline-flex items-center text-gray-500 hover:text-blue-600 mb-6 transition-colors"
                >
                    <ArrowLeft size={16} className="mr-2" />
                    Back to Sessions
                </Link>

                <header className="mb-8 p-6 bg-white rounded-xl shadow-sm border border-gray-100">
                    <div className="flex flex-col md:flex-row justify-between md:items-center gap-4">
                        <div>
                            <h1 className="text-2xl font-bold text-gray-900 mb-2 font-mono">{session.id}</h1>
                            <div className="flex items-center gap-2 text-gray-500 text-sm">
                                <Calendar size={14} />
                                {new Date(session.created_at).toLocaleString()}
                            </div>
                        </div>

                        {session.latency_metrics.average_latency && (
                            <div className="flex items-center gap-3 px-4 py-2 bg-amber-50 text-amber-800 rounded-lg border border-amber-100">
                                <BarChart size={20} />
                                <div>
                                    <div className="text-xs font-semibold uppercase tracking-wider text-amber-600">Avg Latency</div>
                                    <div className="text-lg font-bold">{session.latency_metrics.average_latency.toFixed(2)}s</div>
                                </div>
                            </div>
                        )}
                    </div>
                </header>

                <div className="grid md:grid-cols-3 gap-8">
                    {/* Left Column: Playback & Visualization */}
                    <div className="md:col-span-3 space-y-6">
                        <section className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                            <h2 className="text-lg font-semibold text-gray-900 mb-4">Playback & Latency</h2>
                            <AudioPlayer
                                src={session.audio_url}
                                transcript={session.transcript}
                                freezeEvents={session.freeze_events}
                                sessionStartTime={new Date(session.created_at).getTime() / 1000}
                                onTimeUpdate={setCurrentTime}
                            />
                        </section>

                        <section className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                            <h2 className="text-lg font-semibold text-gray-900 mb-4">Transcript</h2>
                            <TranscriptViewer
                                transcript={session.transcript}
                                sessionStartTime={new Date(session.created_at).getTime() / 1000}
                                currentTime={currentTime}
                            />
                        </section>
                    </div>
                </div>
            </div>
        </main>
    );
}
