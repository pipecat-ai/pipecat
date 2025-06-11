'use client';

import { PipecatClient } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';
import { PipecatClientProvider } from '@pipecat-ai/client-react';
import { PropsWithChildren, useEffect, useState } from 'react';

export function PipecatProvider({ children }: PropsWithChildren) {
  const [client, setClient] = useState<PipecatClient | null>(null);

  useEffect(() => {
    const pcClient = new PipecatClient({
      transport: new DailyTransport(),
      enableMic: true,
      enableCam: false,
    });

    setClient(pcClient);
  }, []);

  if (!client) {
    return null;
  }

  return (
    <PipecatClientProvider client={client}>{children}</PipecatClientProvider>
  );
}
