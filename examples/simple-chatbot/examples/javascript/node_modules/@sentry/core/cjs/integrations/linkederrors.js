Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const integration = require('../integration.js');

const DEFAULT_KEY = 'cause';
const DEFAULT_LIMIT = 5;

const INTEGRATION_NAME = 'LinkedErrors';

const _linkedErrorsIntegration = ((options = {}) => {
  const limit = options.limit || DEFAULT_LIMIT;
  const key = options.key || DEFAULT_KEY;

  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    preprocessEvent(event, hint, client) {
      const options = client.getOptions();

      utils.applyAggregateErrorsToEvent(
        utils.exceptionFromError,
        options.stackParser,
        options.maxValueLength,
        key,
        limit,
        event,
        hint,
      );
    },
  };
}) ;

const linkedErrorsIntegration = integration.defineIntegration(_linkedErrorsIntegration);

/**
 * Adds SDK info to an event.
 * @deprecated Use `linkedErrorsIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const LinkedErrors = integration.convertIntegrationFnToClass(INTEGRATION_NAME, linkedErrorsIntegration)

;

exports.LinkedErrors = LinkedErrors;
exports.linkedErrorsIntegration = linkedErrorsIntegration;
//# sourceMappingURL=linkederrors.js.map
