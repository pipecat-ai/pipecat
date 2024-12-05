import { Event, EventHint } from '@sentry/types';
import { ReplayContainer } from '../types';
/**
 * Returns a listener to be added to `addEventProcessor(listener)`.
 */
export declare function handleGlobalEventListener(replay: ReplayContainer, includeAfterSendEventHandling?: boolean): (event: Event, hint: EventHint) => Event | null;
//# sourceMappingURL=handleGlobalEvent.d.ts.map
