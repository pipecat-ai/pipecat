import type { Transport } from '@sentry/types';
import type { BrowserTransportOptions } from './types';
import type { FetchImpl } from './utils';
/**
 * Creates a Transport that uses the Fetch API to send events to Sentry.
 */
export declare function makeFetchTransport(options: BrowserTransportOptions, nativeFetch?: FetchImpl): Transport;
//# sourceMappingURL=fetch.d.ts.map