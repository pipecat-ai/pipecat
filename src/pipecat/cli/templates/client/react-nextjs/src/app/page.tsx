'use client';

import { useState } from 'react';

import { ThemeProvider } from '@pipecat-ai/voice-ui-kit';

import type { PipecatBaseChildProps } from '@pipecat-ai/voice-ui-kit';
import {
  ErrorCard,
  FullScreenContainer,
  PipecatAppBase,
  SpinLoader,
} from '@pipecat-ai/voice-ui-kit';

import { App } from './components/App';
import {
  AVAILABLE_TRANSPORTS,
  DEFAULT_TRANSPORT,
  TRANSPORT_PROPS,
} from '../config';
import type { TransportType } from '../config';

export default function Home() {
  const [transportType, setTransportType] =
    useState<TransportType>(DEFAULT_TRANSPORT);

  const transportProps = TRANSPORT_PROPS[transportType];

  return (
    <ThemeProvider defaultTheme="terminal" disableStorage>
      <FullScreenContainer>
        <PipecatAppBase
          {...transportProps}
          transportType={transportType}>
          {({
            client,
            handleConnect,
            handleDisconnect,
            error,
          }: PipecatBaseChildProps) =>
            !client ? (
              <SpinLoader />
            ) : error ? (
              <ErrorCard>{error}</ErrorCard>
            ) : (
              <App
                client={client}
                handleConnect={handleConnect}
                handleDisconnect={handleDisconnect}
                transportType={transportType}
                onTransportChange={setTransportType}
                availableTransports={AVAILABLE_TRANSPORTS}
              />
            )
          }
        </PipecatAppBase>
      </FullScreenContainer>
    </ThemeProvider>
  );
}
