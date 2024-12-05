import { type PropsWithChildren } from 'react';
import { RTVIClient } from 'realtime-ai';
import { DailyTransport } from '@daily-co/realtime-ai-daily';
import { RTVIClientProvider } from 'realtime-ai-react';

const transport = new DailyTransport();

const client = new RTVIClient({
  transport,
  params: {
    baseUrl: 'http://localhost:7860',
    endpoints: {
      connect: '/connect',
    },
  },
  enableMic: true,
  enableCam: false,
});

export function RTVIProvider({ children }: PropsWithChildren) {
  return <RTVIClientProvider client={client}>{children}</RTVIClientProvider>;
}
