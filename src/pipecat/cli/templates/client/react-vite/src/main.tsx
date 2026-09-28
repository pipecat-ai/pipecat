import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';

import { Console } from '@/components/pipecat/console/console';

import { TransportSelect } from './components/TransportSelect';
import {
  AVAILABLE_TRANSPORTS,
  DEFAULT_TRANSPORT,
  PROJECT_NAME,
  TRANSPORT_FACTORIES,
  TRANSPORT_PROPS,
} from './config';
import type { TransportType } from './config';
import './index.css';

/**
 * The Pipecat UI console: connect flow, transcript, metrics, device and
 * session info, and a live event stream. It is composed from the components
 * under src/components/pipecat, which are yours to build a custom UI from;
 * the README shows the minimal composition.
 */
export const Main = () => {
  const [transportType, setTransportType] =
    useState<TransportType>(DEFAULT_TRANSPORT);

  return (
    <div className="h-dvh">
      <Console
        // The console reads its transport factory once; remount to switch.
        key={transportType}
        transportType={transportType}
        transportFactory={TRANSPORT_FACTORIES[transportType]}
        {...TRANSPORT_PROPS[transportType]}
        titleText={PROJECT_NAME}
        headerSlot={
          AVAILABLE_TRANSPORTS.length > 1 ? (
            <TransportSelect
              transportType={transportType}
              onTransportChange={setTransportType}
              availableTransports={AVAILABLE_TRANSPORTS}
            />
          ) : undefined
        }
      />
    </div>
  );
};

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Main />
  </StrictMode>
);
