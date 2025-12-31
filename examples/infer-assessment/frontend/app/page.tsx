"use client";

import { useEffect, useState } from "react";
import { fetchSessions, Session } from "../lib/api";
import Link from "next/link";
import { ArrowRight, Clock, Activity, MessageSquare } from "lucide-react";

export default function Home() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSessions()
      .then((data) => {
        setSessions(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  return (
    <main className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <header className="mb-10">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Conversation Recordings</h1>
          <p className="text-gray-500">View transcripts and replay conversations with AI agents.</p>
        </header>

        {loading ? (
          <div className="flex justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        ) : sessions.length === 0 ? (
          <div className="text-center p-12 bg-white rounded-lg shadow-sm border border-gray-100">
            <p className="text-gray-500">No sessions recorded yet.</p>
          </div>
        ) : (
          <div className="grid gap-4">
            {sessions.map((session) => (
              <Link
                href={`/sessions/${session.id}`}
                key={session.id}
                className="block bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md hover:border-blue-100 transition-all group"
              >
                <div className="flex justify-between items-center">
                  <div>
                    <h2 className="text-lg font-semibold text-gray-900 mb-1 font-mono">
                      {session.id.slice(0, 8)}...
                    </h2>
                    <div className="flex gap-4 text-sm text-gray-500">
                      <div className="flex items-center gap-1">
                        <Clock size={14} />
                        {new Date(session.created_at).toLocaleString()}
                      </div>
                      <div className="flex items-center gap-1">
                        <MessageSquare size={14} />
                        {session.transcript.length} turns
                      </div>
                      {session.latency_metrics.average_latency && (
                        <div className="flex items-center gap-1 text-amber-600">
                          <Activity size={14} />
                          Avg Latency: {session.latency_metrics.average_latency.toFixed(2)}s
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="text-blue-600 opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 transition-all duration-300">
                    <ArrowRight size={20} />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
