Object.defineProperty(exports, '__esModule', { value: true });

const core = require('@sentry/core');

const INTEGRATION_NAME = 'SessionTiming';

const _sessionTimingIntegration = (() => {
  const startTime = Date.now();

  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    processEvent(event) {
      const now = Date.now();

      return {
        ...event,
        extra: {
          ...event.extra,
          ['session:start']: startTime,
          ['session:duration']: now - startTime,
          ['session:end']: now,
        },
      };
    },
  };
}) ;

const sessionTimingIntegration = core.defineIntegration(_sessionTimingIntegration);

/**
 * This function adds duration since Sentry was initialized till the time event was sent.
 * @deprecated Use `sessionTimingIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const SessionTiming = core.convertIntegrationFnToClass(
  INTEGRATION_NAME,
  sessionTimingIntegration,
) ;

exports.SessionTiming = SessionTiming;
exports.sessionTimingIntegration = sessionTimingIntegration;
//# sourceMappingURL=sessiontiming.js.map
