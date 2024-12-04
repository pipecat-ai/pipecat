import { DEBUG_BUILD } from '../debug-build.js';
import { logger } from '../logger.js';
import { addConsoleInstrumentationHandler } from './console.js';
export { addConsoleInstrumentationHandler } from './console.js';
import { addClickKeypressInstrumentationHandler } from './dom.js';
export { addClickKeypressInstrumentationHandler } from './dom.js';
import { addFetchInstrumentationHandler } from './fetch.js';
export { addFetchInstrumentationHandler } from './fetch.js';
import { addGlobalErrorInstrumentationHandler } from './globalError.js';
export { addGlobalErrorInstrumentationHandler } from './globalError.js';
import { addGlobalUnhandledRejectionInstrumentationHandler } from './globalUnhandledRejection.js';
export { addGlobalUnhandledRejectionInstrumentationHandler } from './globalUnhandledRejection.js';
import { addHistoryInstrumentationHandler } from './history.js';
export { addHistoryInstrumentationHandler } from './history.js';
import { addXhrInstrumentationHandler } from './xhr.js';
export { SENTRY_XHR_DATA_KEY, addXhrInstrumentationHandler } from './xhr.js';

// TODO(v8): Consider moving this file (or at least parts of it) into the browser package. The registered handlers are mostly non-generic and we risk leaking runtime specific code into generic packages.

/**
 * Add handler that will be called when given type of instrumentation triggers.
 * Use at your own risk, this might break without changelog notice, only used internally.
 * @hidden
 * @deprecated Use the proper function per instrumentation type instead!
 */
function addInstrumentationHandler(type, callback) {
  switch (type) {
    case 'console':
      return addConsoleInstrumentationHandler(callback);
    case 'dom':
      return addClickKeypressInstrumentationHandler(callback);
    case 'xhr':
      return addXhrInstrumentationHandler(callback);
    case 'fetch':
      return addFetchInstrumentationHandler(callback);
    case 'history':
      return addHistoryInstrumentationHandler(callback);
    case 'error':
      return addGlobalErrorInstrumentationHandler(callback);
    case 'unhandledrejection':
      return addGlobalUnhandledRejectionInstrumentationHandler(callback);
    default:
      DEBUG_BUILD && logger.warn('unknown instrumentation type:', type);
  }
}

export { addInstrumentationHandler };
//# sourceMappingURL=index.js.map
