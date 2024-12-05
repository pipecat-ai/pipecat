import type { Event, EventProcessor, Hub, Integration } from '@sentry/types';
type LocalForage = {
    setItem<T>(key: string, value: T, callback?: (err: any, value: T) => void): Promise<T>;
    iterate<T, U>(iteratee: (value: T, key: string, iterationNumber: number) => U, callback?: (err: any, result: U) => void): Promise<U>;
    removeItem(key: string, callback?: (err: any) => void): Promise<void>;
    length(): Promise<number>;
};
export type Item = {
    key: string;
    value: Event;
};
/**
 * cache offline errors and send when connected
 * @deprecated The offline integration has been deprecated in favor of the offline transport wrapper.
 *
 * http://docs.sentry.io/platforms/javascript/configuration/transports/#offline-caching
 */
export declare class Offline implements Integration {
    /**
     * @inheritDoc
     */
    static id: string;
    /**
     * @inheritDoc
     */
    readonly name: string;
    /**
     * the current hub instance
     */
    hub?: Hub;
    /**
     * maximum number of events to store while offline
     */
    maxStoredEvents: number;
    /**
     * event cache
     */
    offlineEventStore: LocalForage;
    /**
     * @inheritDoc
     */
    constructor(options?: {
        maxStoredEvents?: number;
    });
    /**
     * @inheritDoc
     */
    setupOnce(addGlobalEventProcessor: (callback: EventProcessor) => void, getCurrentHub: () => Hub): void;
    /**
     * cache an event to send later
     * @param event an event
     */
    private _cacheEvent;
    /**
     * purge excess events if necessary
     */
    private _enforceMaxEvents;
    /**
     * purge event from cache
     */
    private _purgeEvent;
    /**
     * purge events from cache
     */
    private _purgeEvents;
    /**
     * send all events
     */
    private _sendEvents;
}
export {};
//# sourceMappingURL=offline.d.ts.map