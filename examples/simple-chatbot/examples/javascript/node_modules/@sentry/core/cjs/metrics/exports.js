Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const debugBuild = require('../debug-build.js');
const exports$1 = require('../exports.js');
const spanUtils = require('../utils/spanUtils.js');
const constants = require('./constants.js');
const integration = require('./integration.js');

function addToMetricsAggregator(
  metricType,
  name,
  value,
  data = {},
) {
  const client = exports$1.getClient();
  const scope = exports$1.getCurrentScope();
  if (client) {
    if (!client.metricsAggregator) {
      debugBuild.DEBUG_BUILD &&
        utils.logger.warn('No metrics aggregator enabled. Please add the MetricsAggregator integration to use metrics APIs');
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
      metricTags.transaction = spanUtils.spanToJSON(transaction).description || '';
    }

    debugBuild.DEBUG_BUILD && utils.logger.log(`Adding value of ${value} to ${metricType} metric ${name}`);
    client.metricsAggregator.add(metricType, name, value, unit, { ...metricTags, ...tags }, timestamp);
  }
}

/**
 * Adds a value to a counter metric
 *
 * @experimental This API is experimental and might have breaking changes in the future.
 */
function increment(name, value = 1, data) {
  addToMetricsAggregator(constants.COUNTER_METRIC_TYPE, name, value, data);
}

/**
 * Adds a value to a distribution metric
 *
 * @experimental This API is experimental and might have breaking changes in the future.
 */
function distribution(name, value, data) {
  addToMetricsAggregator(constants.DISTRIBUTION_METRIC_TYPE, name, value, data);
}

/**
 * Adds a value to a set metric. Value must be a string or integer.
 *
 * @experimental This API is experimental and might have breaking changes in the future.
 */
function set(name, value, data) {
  addToMetricsAggregator(constants.SET_METRIC_TYPE, name, value, data);
}

/**
 * Adds a value to a gauge metric
 *
 * @experimental This API is experimental and might have breaking changes in the future.
 */
function gauge(name, value, data) {
  addToMetricsAggregator(constants.GAUGE_METRIC_TYPE, name, value, data);
}

const metrics = {
  increment,
  distribution,
  set,
  gauge,
  /** @deprecated Use `metrics.metricsAggregratorIntegration()` instead. */
  // eslint-disable-next-line deprecation/deprecation
  MetricsAggregator: integration.MetricsAggregator,
  metricsAggregatorIntegration: integration.metricsAggregatorIntegration,
};

exports.distribution = distribution;
exports.gauge = gauge;
exports.increment = increment;
exports.metrics = metrics;
exports.set = set;
//# sourceMappingURL=exports.js.map
