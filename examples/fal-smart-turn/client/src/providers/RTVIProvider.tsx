'use client';

import { RTVIClient } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';
import { RTVIClientProvider } from '@pipecat-ai/client-react';
import { PropsWithChildren, useEffect, useState } from 'react';

// Get the API base URL from environment variables
// Default to "/api" if not specified
// "/api" is the default for Next.js API routes and used
// for the Pipecat Cloud deployed agent
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '/api';

console.log('Using API base URL:', API_BASE_URL);

export function RTVIProvider({ children }: PropsWithChildren) {
  const [client, setClient] = useState<RTVIClient | null>(null);

  useEffect(() => {
    const transport = new DailyTransport();

    const rtviClient = new RTVIClient({
      transport,
      params: {
        baseUrl: API_BASE_URL,
        endpoints: {
          connect: '/connect',
        },
        requestData: { foo: 'bar' },
      },
      enableMic: true,
      enableCam: false,
    });

    setClient(rtviClient);
  }, []);

  if (!client) {
    return null;
  }

  return <RTVIClientProvider client={client}>{children}</RTVIClientProvider>;
}
