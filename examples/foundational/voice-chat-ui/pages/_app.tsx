// pages/_app.tsx

import '../styles/globals.css';
import type { AppProps } from 'next/app';
import { useEffect, useState } from 'react';

import { RTVIClient } from '@pipecat-ai/client-js';
import { RTVIClientProvider, RTVIClientAudio } from '@pipecat-ai/client-react';
import { SmallWebRTCTransport } from '@pipecat-ai/small-webrtc-transport';

// --- Configuration ---
// Use your Cloudflare tunnel URL or localhost for local development.
// IMPORTANT: Ensure your backend CORS is configured to allow this origin.
// The backend `43-ollama-chatbot_rag.py` has no CORS by default, but the
// pipecat library adds it for common dev origins. If your public URL
// doesn't work, this is a likely culprit.
const BASE_URL = process.env.NEXT_PUBLIC_PIPECAT_URL || 'https://ai.alexcovo.com';
const WS_URL = BASE_URL.replace(/^http/, 'ws') + '/ws';

export default function MyApp({ Component, pageProps }: AppProps) {
  const [client, setClient] = useState<RTVIClient | null>(null);

  useEffect(() => {
    if (client || typeof window === 'undefined') {
      return;
    }

    console.log(`[App] Initializing Pipecat client for ${BASE_URL}`);

    // SmallWebRTCTransport doesn't need arguments here;
    // it will be configured by the RTVIClient's params.
    const transport = new SmallWebRTCTransport();

    const newClient = new RTVIClient({
      transport,
      params: {
        // Base URL for the HTTP signaling server (POST /api/offer)
        baseUrl: BASE_URL,
        // URL for the WebRTC WebSocket connection
        wsUrl: WS_URL,
        // THIS IS THE FIX: Explicitly tell the client to use /api/offer
        endpoints: {
          connect: '/api/offer',
        },
      },
      enableMic: true,
    });

    setClient(newClient);

    // No cleanup needed here as the client object is managed in state
    // and passed to the provider.
  }, [client]);

  // Render a loading state or null until the client is initialized on the browser.
  if (!client) {
    return <div>Loading Assistant...</div>;
  }

  return (
    <RTVIClientProvider client={client}>
      <Component {...pageProps} />
      <RTVIClientAudio />
    </RTVIClientProvider>
  );
}