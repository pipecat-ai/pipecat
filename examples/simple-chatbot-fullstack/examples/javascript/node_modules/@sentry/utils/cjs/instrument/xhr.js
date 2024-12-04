Object.defineProperty(exports, '__esModule', { value: true });

const is = require('../is.js');
const object = require('../object.js');
const worldwide = require('../worldwide.js');
const _handlers = require('./_handlers.js');

const WINDOW = worldwide.GLOBAL_OBJ ;

const SENTRY_XHR_DATA_KEY = '__sentry_xhr_v3__';

/**
 * Add an instrumentation handler for when an XHR request happens.
 * The handler function is called once when the request starts and once when it ends,
 * which can be identified by checking if it has an `endTimestamp`.
 *
 * Use at your own risk, this might break without changelog notice, only used internally.
 * @hidden
 */
function addXhrInstrumentationHandler(handler) {
  const type = 'xhr';
  _handlers.addHandler(type, handler);
  _handlers.maybeInstrument(type, instrumentXHR);
}

/** Exported only for tests. */
function instrumentXHR() {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  if (!(WINDOW ).XMLHttpRequest) {
    return;
  }

  const xhrproto = XMLHttpRequest.prototype;

  object.fill(xhrproto, 'open', function (originalOpen) {
    return function ( ...args) {
      const startTimestamp = Date.now();

      // open() should always be called with two or more arguments
      // But to be on the safe side, we actually validate this and bail out if we don't have a method & url
      const method = is.isString(args[0]) ? args[0].toUpperCase() : undefined;
      const url = parseUrl(args[1]);

      if (!method || !url) {
        return originalOpen.apply(this, args);
      }

      this[SENTRY_XHR_DATA_KEY] = {
        method,
        url,
        request_headers: {},
      };

      // if Sentry key appears in URL, don't capture it as a request
      if (method === 'POST' && url.match(/sentry_key/)) {
        this.__sentry_own_request__ = true;
      }

      const onreadystatechangeHandler = () => {
        // For whatever reason, this is not the same instance here as from the outer method
        const xhrInfo = this[SENTRY_XHR_DATA_KEY];

        if (!xhrInfo) {
          return;
        }

        if (this.readyState === 4) {
          try {
            // touching statusCode in some platforms throws
            // an exception
            xhrInfo.status_code = this.status;
          } catch (e) {
            /* do nothing */
          }

          const handlerData = {
            args: [method, url],
            endTimestamp: Date.now(),
            startTimestamp,
            xhr: this,
          };
          _handlers.triggerHandlers('xhr', handlerData);
        }
      };

      if ('onreadystatechange' in this && typeof this.onreadystatechange === 'function') {
        object.fill(this, 'onreadystatechange', function (original) {
          return function ( ...readyStateArgs) {
            onreadystatechangeHandler();
            return original.apply(this, readyStateArgs);
          };
        });
      } else {
        this.addEventListener('readystatechange', onreadystatechangeHandler);
      }

      // Intercepting `setRequestHeader` to access the request headers of XHR instance.
      // This will only work for user/library defined headers, not for the default/browser-assigned headers.
      // Request cookies are also unavailable for XHR, as `Cookie` header can't be defined by `setRequestHeader`.
      object.fill(this, 'setRequestHeader', function (original) {
        return function ( ...setRequestHeaderArgs) {
          const [header, value] = setRequestHeaderArgs;

          const xhrInfo = this[SENTRY_XHR_DATA_KEY];

          if (xhrInfo && is.isString(header) && is.isString(value)) {
            xhrInfo.request_headers[header.toLowerCase()] = value;
          }

          return original.apply(this, setRequestHeaderArgs);
        };
      });

      return originalOpen.apply(this, args);
    };
  });

  object.fill(xhrproto, 'send', function (originalSend) {
    return function ( ...args) {
      const sentryXhrData = this[SENTRY_XHR_DATA_KEY];

      if (!sentryXhrData) {
        return originalSend.apply(this, args);
      }

      if (args[0] !== undefined) {
        sentryXhrData.body = args[0];
      }

      const handlerData = {
        args: [sentryXhrData.method, sentryXhrData.url],
        startTimestamp: Date.now(),
        xhr: this,
      };
      _handlers.triggerHandlers('xhr', handlerData);

      return originalSend.apply(this, args);
    };
  });
}

function parseUrl(url) {
  if (is.isString(url)) {
    return url;
  }

  try {
    // url can be a string or URL
    // but since URL is not available in IE11, we do not check for it,
    // but simply assume it is an URL and return `toString()` from it (which returns the full URL)
    // If that fails, we just return undefined
    return (url ).toString();
  } catch (e2) {} // eslint-disable-line no-empty

  return undefined;
}

exports.SENTRY_XHR_DATA_KEY = SENTRY_XHR_DATA_KEY;
exports.addXhrInstrumentationHandler = addXhrInstrumentationHandler;
exports.instrumentXHR = instrumentXHR;
//# sourceMappingURL=xhr.js.map
