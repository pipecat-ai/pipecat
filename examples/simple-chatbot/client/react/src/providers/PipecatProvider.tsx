import { type PropsWithChildren } from 'react';
import { PipecatClient } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';
import { PipecatClientProvider } from '@pipecat-ai/client-react';

const client = new PipecatClient({
  transport: new DailyTransport(),
  enableMic: true,
  enableCam: false,
});

export function PipecatProvider({ children }: PropsWithChildren) {
  return (
    <PipecatClientProvider client={client}>{children}</PipecatClientProvider>
  );
}
