'use client';

import { useState } from 'react';

import { Console } from '@/components/pipecat/console/console';
import {
  AVAILABLE_TRANSPORTS,
  DEFAULT_TRANSPORT,
  PROJECT_NAME,
  TRANSPORT_FACTORIES,
  TRANSPORT_PROPS,
} from '@/config';
import type { TransportType } from '@/config';

import { HeaderBrand } from './components/HeaderBrand';
import { TransportSelect } from './components/TransportSelect';

/**
 * The Pipecat UI console: connect flow, transcript, metrics, device and
 * session info, and a live event stream. It is composed from the components
 * under src/components/pipecat, which are yours to build a custom UI from;
 * the README shows the minimal composition.
 */
export default function Home() {
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
        // The logo slot is the header's leftmost element: the transport picker
        // goes there, and the brand centers itself against the console root.
        className="relative"
        titleText=""
        logo={
          <>
            {AVAILABLE_TRANSPORTS.length > 1 && (
              <TransportSelect
                transportType={transportType}
                onTransportChange={setTransportType}
                availableTransports={AVAILABLE_TRANSPORTS}
              />
            )}
            <HeaderBrand name={PROJECT_NAME} />
          </>
        }
      />
    </div>
  );
}
