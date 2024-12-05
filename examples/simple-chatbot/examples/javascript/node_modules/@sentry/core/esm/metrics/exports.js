import { logger } from '@sentry/utils';
import { DEBUG_BUILD } from '../debug-build.js';
import { getClient, getCurrentScope } from '../exports.js';
import { spanToJSON } from '../utils/spanUtils.js';
import { COUNTER_METRIC_TYPE, DISTRIBUTION_METRIC_TYPE, SET_METRIC_TYPE, GAUGE_METRIC_TYPE } from './constants.js';
import { MetricsAggregator, metricsAggregatorIntegration } from './integration.js';

function addToMetricsAggregator(
  metricType,
  name,
  value,
  data = {},
) {
  const client = getClient();
  const scope = getCurrentScope();
  if (client) {
    if (!client.metricsAggregator) {
      DEBUG_BUILD &&
        logger.warn('No metrics aggregator enabled. Please add the MetricsAggregator integration to use metrics APIs');
      return;
    }
    const { unit, tags, timestamp } = data;
    const { release, environment } = client.getOptions();
    // eslint-disable-next-line deprecation/deprecation
    const transaction = scope.getTransaction();
    const metricTags = {};
    if (release) {
      metricTags.release = release;
    }
    if (environment) {
      metricTags.environment = environment;
    }
    if (transaction) {
      metricTags.transaction = spanToJSON(transaction).description || '';
    }

    DEBUG_BUILD && logger.log(`Adding value of ${value} to ${metricType} metric ${name}`);
    client.metricsAggregator.add(metricType, name, value, unit, { ...metricTags, ...tags }, timestamp);
  }
}

/**
 * Adds a value to a counter metric
 *
 * @experimental This API is experimental and might have breaking changes in the future.
 */
function increment(name, value = 1, data) {
  addToMetricsAggregator(COUNTER_METRIC_TYPE, name, value, data);
}

/**
 * Adds a value to a distribution metric
 *
 * @experimental This API is experimental and might have breaking changes in the future.
 */
function distribution(name, value, data) {
  addToMetricsAggregator(DISTRIBUTION_METRIC_TYPE, name, value, data);
}

/**
 * Adds a value to a set metric. Value must be a string or integer.
 *
 * @experimental This API is experimental and might have breaking changes in the future.
 */
function set(name, value, data) {
  addToMetricsAggregator(SET_METRIC_TYPE, name, value, data);
}

/**
 * Adds a value to a gauge metric
 *
 * @experimental This API is experimental and might have breaking changes in the future.
 */
function gauge(name, value, data) {
  addToMetricsAggregator(GAUGE_METRIC_TYPE, name, value, data);
}

const metrics = {
  increment,
  distribution,
  set,
  gauge,
  /** @deprecated Use `metrics.metricsAggregratorIntegration()` instead. */
  // eslint-disable-next-line deprecation/deprecation
  MetricsAggregator,
  metricsAggregatorIntegration,
};

export { distribution, gauge, increment, metrics, set };
//# sourceMappingURL=exports.js.map
