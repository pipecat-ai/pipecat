Object.defineProperty(exports, '__esModule', { value: true });

const core = require('@sentry/core');
const helpers = require('../helpers.js');

const INTEGRATION_NAME = 'HttpContext';

const _httpContextIntegration = (() => {
  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    preprocessEvent(event) {
      // if none of the information we want exists, don't bother
      if (!helpers.WINDOW.navigator && !helpers.WINDOW.location && !helpers.WINDOW.document) {
        return;
      }

      // grab as much info as exists and add it to the event
      const url = (event.request && event.request.url) || (helpers.WINDOW.location && helpers.WINDOW.location.href);
      const { referrer } = helpers.WINDOW.document || {};
      const { userAgent } = helpers.WINDOW.navigator || {};

      const headers = {
        ...(event.request && event.request.headers),
        ...(referrer && { Referer: referrer }),
        ...(userAgent && { 'User-Agent': userAgent }),
      };
      const request = { ...event.request, ...(url && { url }), headers };

      event.request = request;
    },
  };
}) ;

const httpContextIntegration = core.defineIntegration(_httpContextIntegration);

/**
 * HttpContext integration collects information about HTTP request headers.
 * @deprecated Use `httpContextIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const HttpContext = core.convertIntegrationFnToClass(INTEGRATION_NAME, httpContextIntegration)

;

exports.HttpContext = HttpContext;
exports.httpContextIntegration = httpContextIntegration;
//# sourceMappingURL=httpcontext.js.map
