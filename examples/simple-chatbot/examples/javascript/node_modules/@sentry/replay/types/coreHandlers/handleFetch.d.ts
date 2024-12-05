import type { HandlerDataFetch } from '@sentry/types';
import type { NetworkRequestData, ReplayContainer, ReplayPerformanceEntry } from '../types';
/** only exported for tests */
export declare function handleFetch(handlerData: HandlerDataFetch): null | ReplayPerformanceEntry<NetworkRequestData>;
/**
 * Returns a listener to be added to `addFetchInstrumentationHandler(listener)`.
 */
export declare function handleFetchSpanListener(replay: ReplayContainer): (handlerData: HandlerDataFetch) => void;
//# sourceMappingURL=handleFetch.d.ts.map