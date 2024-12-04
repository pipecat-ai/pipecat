import type { HandlerDataXhr } from '@sentry/types';
import type { NetworkRequestData, ReplayContainer, ReplayPerformanceEntry } from '../types';
/** only exported for tests */
export declare function handleXhr(handlerData: HandlerDataXhr): ReplayPerformanceEntry<NetworkRequestData> | null;
/**
 * Returns a listener to be added to `addXhrInstrumentationHandler(listener)`.
 */
export declare function handleXhrSpanListener(replay: ReplayContainer): (handlerData: HandlerDataXhr) => void;
//# sourceMappingURL=handleXhr.d.ts.map