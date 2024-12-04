Object.defineProperty(exports, '__esModule', { value: true });

const object = require('../object.js');
const supports = require('../supports.js');
const worldwide = require('../worldwide.js');
const _handlers = require('./_handlers.js');

/**
 * Add an instrumentation handler for when a fetch request happens.
 * The handler function is called once when the request starts and once when it ends,
 * which can be identified by checking if it has an `endTimestamp`.
 *
 * Use at your own risk, this might break without changelog notice, only used internally.
 * @hidden
 */
function addFetchInstrumentationHandler(handler) {
  const type = 'fetch';
  _handlers.addHandler(type, handler);
  _handlers.maybeInstrument(type, instrumentFetch);
}

function instrumentFetch() {
  if (!supports.supportsNativeFetch()) {
    return;
  }

  object.fill(worldwide.GLOBAL_OBJ, 'fetch', function (originalFetch) {
    return function (...args) {
      const { method, url } = parseFetchArgs(args);

      const handlerData = {
        args,
        fetchData: {
          method,
          url,
        },
        startTimestamp: Date.now(),
      };

      _handlers.triggerHandlers('fetch', {
        ...handlerData,
      });

      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      return originalFetch.apply(worldwide.GLOBAL_OBJ, args).then(
        (response) => {
          const finishedHandlerData = {
            ...handlerData,
            endTimestamp: Date.now(),
            response,
          };

          _handlers.triggerHandlers('fetch', finishedHandlerData);
          return response;
        },
        (error) => {
          const erroredHandlerData = {
            ...handlerData,
            endTimestamp: Date.now(),
            error,
          };

          _handlers.triggerHandlers('fetch', erroredHandlerData);
          // NOTE: If you are a Sentry user, and you are seeing this stack frame,
          //       it means the sentry.javascript SDK caught an error invoking your application code.
          //       This is expected behavior and NOT indicative of a bug with sentry.javascript.
          throw error;
        },
      );
    };
  });
}

function hasProp(obj, prop) {
  return !!obj && typeof obj === 'object' && !!(obj )[prop];
}

function getUrlFromResource(resource) {
  if (typeof resource === 'string') {
    return resource;
  }

  if (!resource) {
    return '';
  }

  if (hasProp(resource, 'url')) {
    return resource.url;
  }

  if (resource.toString) {
    return resource.toString();
  }

  return '';
}

/**
 * Parses the fetch arguments to find the used Http method and the url of the request.
 * Exported for tests only.
 */
function parseFetchArgs(fetchArgs) {
  if (fetchArgs.length === 0) {
    return { method: 'GET', url: '' };
  }

  if (fetchArgs.length === 2) {
    const [url, options] = fetchArgs ;

    return {
      url: getUrlFromResource(url),
      method: hasProp(options, 'method') ? String(options.method).toUpperCase() : 'GET',
    };
  }

  const arg = fetchArgs[0];
  return {
    url: getUrlFromResource(arg ),
    method: hasProp(arg, 'method') ? String(arg.method).toUpperCase() : 'GET',
  };
}

exports.addFetchInstrumentationHandler = addFetchInstrumentationHandler;
exports.parseFetchArgs = parseFetchArgs;
//# sourceMappingURL=fetch.js.map
