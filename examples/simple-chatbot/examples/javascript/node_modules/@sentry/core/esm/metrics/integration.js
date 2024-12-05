import { convertIntegrationFnToClass, defineIntegration } from '../integration.js';
import { BrowserMetricsAggregator } from './browser-aggregator.js';

const INTEGRATION_NAME = 'MetricsAggregator';

const _metricsAggregatorIntegration = (() => {
  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    setup(client) {
      client.metricsAggregator = new BrowserMetricsAggregator(client);
    },
  };
}) ;

const metricsAggregatorIntegration = defineIntegration(_metricsAggregatorIntegration);

/**
 * Enables Sentry metrics monitoring.
 *
 * @experimental This API is experimental and might having breaking changes in the future.
 * @deprecated Use `metricsAggegratorIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const MetricsAggregator = convertIntegrationFnToClass(
  INTEGRATION_NAME,
  metricsAggregatorIntegration,
) ;

export { MetricsAggregator, metricsAggregatorIntegration };
//# sourceMappingURL=integration.js.map
