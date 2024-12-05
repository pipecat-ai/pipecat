Object.defineProperty(exports, '__esModule', { value: true });

const debugBuild = require('../debug-build.js');
const logger = require('../logger.js');
const console = require('./console.js');
const dom = require('./dom.js');
const fetch = require('./fetch.js');
const globalError = require('./globalError.js');
const globalUnhandledRejection = require('./globalUnhandledRejection.js');
const history = require('./history.js');
const xhr = require('./xhr.js');

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
      return console.addConsoleInstrumentationHandler(callback);
    case 'dom':
      return dom.addClickKeypressInstrumentationHandler(callback);
    case 'xhr':
      return xhr.addXhrInstrumentationHandler(callback);
    case 'fetch':
      return fetch.addFetchInstrumentationHandler(callback);
    case 'history':
      return history.addHistoryInstrumentationHandler(callback);
    case 'error':
      return globalError.addGlobalErrorInstrumentationHandler(callback);
    case 'unhandledrejection':
      return globalUnhandledRejection.addGlobalUnhandledRejectionInstrumentationHandler(callback);
    default:
      debugBuild.DEBUG_BUILD && logger.logger.warn('unknown instrumentation type:', type);
  }
}

exports.addConsoleInstrumentationHandler = console.addConsoleInstrumentationHandler;
exports.addClickKeypressInstrumentationHandler = dom.addClickKeypressInstrumentationHandler;
exports.addFetchInstrumentationHandler = fetch.addFetchInstrumentationHandler;
exports.addGlobalErrorInstrumentationHandler = globalError.addGlobalErrorInstrumentationHandler;
exports.addGlobalUnhandledRejectionInstrumentationHandler = globalUnhandledRejection.addGlobalUnhandledRejectionInstrumentationHandler;
exports.addHistoryInstrumentationHandler = history.addHistoryInstrumentationHandler;
exports.SENTRY_XHR_DATA_KEY = xhr.SENTRY_XHR_DATA_KEY;
exports.addXhrInstrumentationHandler = xhr.addXhrInstrumentationHandler;
exports.addInstrumentationHandler = addInstrumentationHandler;
//# sourceMappingURL=index.js.map
