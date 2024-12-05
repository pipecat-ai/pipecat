import type { Event, EventHint, EventProcessor } from '@sentry/types';
/**
 * Returns the global event processors.
 * @deprecated Global event processors will be removed in v8.
 */
export declare function getGlobalEventProcessors(): EventProcessor[];
/**
 * Add a EventProcessor to be kept globally.
 * @deprecated Use `addEventProcessor` instead. Global event processors will be removed in v8.
 */
export declare function addGlobalEventProcessor(callback: EventProcessor): void;
/**
 * Process an array of event processors, returning the processed event (or `null` if the event was dropped).
 */
export declare function notifyEventProcessors(processors: EventProcessor[], event: Event | null, hint: EventHint, index?: number): PromiseLike<Event | null>;
//# sourceMappingURL=eventProcessors.d.ts.map