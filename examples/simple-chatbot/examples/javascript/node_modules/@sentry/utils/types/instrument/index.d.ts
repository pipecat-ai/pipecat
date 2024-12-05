import type { InstrumentHandlerCallback as _InstrumentHandlerCallback, InstrumentHandlerType as _InstrumentHandlerType } from './_handlers';
import { resetInstrumentationHandlers } from './_handlers';
import { addConsoleInstrumentationHandler } from './console';
import { addClickKeypressInstrumentationHandler } from './dom';
import { addFetchInstrumentationHandler } from './fetch';
import { addGlobalErrorInstrumentationHandler } from './globalError';
import { addGlobalUnhandledRejectionInstrumentationHandler } from './globalUnhandledRejection';
import { addHistoryInstrumentationHandler } from './history';
import { SENTRY_XHR_DATA_KEY, addXhrInstrumentationHandler } from './xhr';
/**
 * Add handler that will be called when given type of instrumentation triggers.
 * Use at your own risk, this might break without changelog notice, only used internally.
 * @hidden
 * @deprecated Use the proper function per instrumentation type instead!
 */
export declare function addInstrumentationHandler(type: _InstrumentHandlerType, callback: _InstrumentHandlerCallback): void;
/**
 * @deprecated Use the specific handler data types from @sentry/types instead, e.g. HandlerDataFetch, HandlerDataConsole, ...
 */
type InstrumentHandlerCallback = _InstrumentHandlerCallback;
/**
 * @deprecated Use the specific handler functions instead, e.g. addConsoleInstrumentationHandler, ...
 */
type InstrumentHandlerType = _InstrumentHandlerType;
export type { InstrumentHandlerCallback, InstrumentHandlerType };
export { addConsoleInstrumentationHandler, addClickKeypressInstrumentationHandler, addXhrInstrumentationHandler, addFetchInstrumentationHandler, addHistoryInstrumentationHandler, addGlobalErrorInstrumentationHandler, addGlobalUnhandledRejectionInstrumentationHandler, SENTRY_XHR_DATA_KEY, resetInstrumentationHandlers, };
//# sourceMappingURL=index.d.ts.map