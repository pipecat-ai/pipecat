'use client';

import { PipecatClient } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';
import { PipecatClientProvider } from '@pipecat-ai/client-react';
import { PropsWithChildren, useEffect, useState, useRef } from 'react';

export function PipecatProvider({ children }: PropsWithChildren) {
  const [client, setClient] = useState<PipecatClient | null>(null);
  const clientCreated = useRef(false);

  useEffect(() => {
    // Only create the client once
    if (clientCreated.current) return;

    const pcClient = new PipecatClient({
      transport: new DailyTransport(),
      enableMic: true,
      enableCam: false,
    });

    setClient(pcClient);
    clientCreated.current = true;

    // Cleanup when component unmounts
    return () => {
      if (pcClient) {
        pcClient.disconnect().catch((err) => {
          console.error('Error disconnecting client:', err);
        });
      }
      clientCreated.current = false;
    };
  }, []);

  if (!client) {
    return null;
  }

  return (
    <PipecatClientProvider client={client}>{children}</PipecatClientProvider>
  );
}
