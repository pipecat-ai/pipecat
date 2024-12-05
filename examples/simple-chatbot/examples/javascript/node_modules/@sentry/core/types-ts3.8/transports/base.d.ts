import { InternalBaseTransportOptions, Transport, TransportMakeRequestResponse, TransportRequestExecutor } from '@sentry/types';
import { PromiseBuffer } from '@sentry/utils';
export declare const DEFAULT_TRANSPORT_BUFFER_SIZE = 30;
/**
 * Creates an instance of a Sentry `Transport`
 *
 * @param options
 * @param makeRequest
 */
export declare function createTransport(options: InternalBaseTransportOptions, makeRequest: TransportRequestExecutor, buffer?: PromiseBuffer<void | TransportMakeRequestResponse>): Transport;
//# sourceMappingURL=base.d.ts.map
