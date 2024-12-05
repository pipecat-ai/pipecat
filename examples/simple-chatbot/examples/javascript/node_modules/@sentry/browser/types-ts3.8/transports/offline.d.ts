import { OfflineTransportOptions } from '@sentry/core';
import { InternalBaseTransportOptions, Transport } from '@sentry/types';
import { TextDecoderInternal } from '@sentry/utils';
type Store = <T>(callback: (store: IDBObjectStore) => T | PromiseLike<T>) => Promise<T>;
/** Create or open an IndexedDb store */
export declare function createStore(dbName: string, storeName: string): Store;
/** Insert into the store */
export declare function insert(store: Store, value: Uint8Array | string, maxQueueSize: number): Promise<void>;
/** Pop the oldest value from the store */
export declare function pop(store: Store): Promise<Uint8Array | string | undefined>;
export interface BrowserOfflineTransportOptions extends OfflineTransportOptions {
    /**
     * Name of indexedDb database to store envelopes in
     * Default: 'sentry-offline'
     */
    dbName?: string;
    /**
     * Name of indexedDb object store to store envelopes in
     * Default: 'queue'
     */
    storeName?: string;
    /**
     * Maximum number of envelopes to store
     * Default: 30
     */
    maxQueueSize?: number;
    /**
     * Only required for testing on node.js
     * @ignore
     */
    textDecoder?: TextDecoderInternal;
}
/**
 * Creates a transport that uses IndexedDb to store events when offline.
 */
export declare function makeBrowserOfflineTransport<T extends InternalBaseTransportOptions>(createTransport: (options: T) => Transport): (options: T & BrowserOfflineTransportOptions) => Transport;
export {};
//# sourceMappingURL=offline.d.ts.map
