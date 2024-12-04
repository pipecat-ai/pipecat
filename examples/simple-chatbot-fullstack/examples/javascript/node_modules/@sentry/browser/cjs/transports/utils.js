Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const debugBuild = require('../debug-build.js');
const helpers = require('../helpers.js');

let cachedFetchImpl = undefined;

/**
 * A special usecase for incorrectly wrapped Fetch APIs in conjunction with ad-blockers.
 * Whenever someone wraps the Fetch API and returns the wrong promise chain,
 * this chain becomes orphaned and there is no possible way to capture it's rejections
 * other than allowing it bubble up to this very handler. eg.
 *
 * const f = window.fetch;
 * window.fetch = function () {
 *   const p = f.apply(this, arguments);
 *
 *   p.then(function() {
 *     console.log('hi.');
 *   });
 *
 *   return p;
 * }
 *
 * `p.then(function () { ... })` is producing a completely separate promise chain,
 * however, what's returned is `p` - the result of original `fetch` call.
 *
 * This mean, that whenever we use the Fetch API to send our own requests, _and_
 * some ad-blocker blocks it, this orphaned chain will _always_ reject,
 * effectively causing another event to be captured.
 * This makes a whole process become an infinite loop, which we need to somehow
 * deal with, and break it in one way or another.
 *
 * To deal with this issue, we are making sure that we _always_ use the real
 * browser Fetch API, instead of relying on what `window.fetch` exposes.
 * The only downside to this would be missing our own requests as breadcrumbs,
 * but because we are already not doing this, it should be just fine.
 *
 * Possible failed fetch error messages per-browser:
 *
 * Chrome:  Failed to fetch
 * Edge:    Failed to Fetch
 * Firefox: NetworkError when attempting to fetch resource
 * Safari:  resource blocked by content blocker
 */
function getNativeFetchImplementation() {
  if (cachedFetchImpl) {
    return cachedFetchImpl;
  }

  /* eslint-disable @typescript-eslint/unbound-method */

  // Fast path to avoid DOM I/O
  if (utils.isNativeFetch(helpers.WINDOW.fetch)) {
    return (cachedFetchImpl = helpers.WINDOW.fetch.bind(helpers.WINDOW));
  }

  const document = helpers.WINDOW.document;
  let fetchImpl = helpers.WINDOW.fetch;
  // eslint-disable-next-line deprecation/deprecation
  if (document && typeof document.createElement === 'function') {
    try {
      const sandbox = document.createElement('iframe');
      sandbox.hidden = true;
      document.head.appendChild(sandbox);
      const contentWindow = sandbox.contentWindow;
      if (contentWindow && contentWindow.fetch) {
        fetchImpl = contentWindow.fetch;
      }
      document.head.removeChild(sandbox);
    } catch (e) {
      debugBuild.DEBUG_BUILD && utils.logger.warn('Could not create sandbox iframe for pure fetch check, bailing to window.fetch: ', e);
    }
  }

  return (cachedFetchImpl = fetchImpl.bind(helpers.WINDOW));
  /* eslint-enable @typescript-eslint/unbound-method */
}

/** Clears cached fetch impl */
function clearCachedFetchImplementation() {
  cachedFetchImpl = undefined;
}

exports.clearCachedFetchImplementation = clearCachedFetchImplementation;
exports.getNativeFetchImplementation = getNativeFetchImplementation;
//# sourceMappingURL=utils.js.map
