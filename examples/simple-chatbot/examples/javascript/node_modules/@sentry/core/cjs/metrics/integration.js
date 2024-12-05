Object.defineProperty(exports, '__esModule', { value: true });

const integration = require('../integration.js');
const browserAggregator = require('./browser-aggregator.js');

const INTEGRATION_NAME = 'MetricsAggregator';

const _metricsAggregatorIntegration = (() => {
  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    setup(client) {
      client.metricsAggregator = new browserAggregator.BrowserMetricsAggregator(client);
    },
  };
}) ;

const metricsAggregatorIntegration = integration.defineIntegration(_metricsAggregatorIntegration);

/**
 * Enables Sentry metrics monitoring.
 *
 * @experimental This API is experimental and might having breaking changes in the future.
 * @deprecated Use `metricsAggegratorIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const MetricsAggregator = integration.convertIntegrationFnToClass(
  INTEGRATION_NAME,
  metricsAggregatorIntegration,
) ;

exports.MetricsAggregator = MetricsAggregator;
exports.metricsAggregatorIntegration = metricsAggregatorIntegration;
//# sourceMappingURL=integration.js.map
