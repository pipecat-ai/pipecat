import type { BaseTransportOptions } from '@sentry/types';
import type { BrowserOfflineTransportOptions } from './offline';
type BaseTransportAndOfflineTransportOptions = BaseTransportOptions & BrowserOfflineTransportOptions;
export interface BrowserTransportOptions extends BaseTransportAndOfflineTransportOptions {
    /** Fetch API init parameters. Used by the FetchTransport */
    fetchOptions?: RequestInit;
    /** Custom headers for the transport. Used by the XHRTransport and FetchTransport */
    headers?: {
        [key: string]: string;
    };
}
export {};
//# sourceMappingURL=types.d.ts.map