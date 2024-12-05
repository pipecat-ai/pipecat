Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
require('../debug-build.js');
require('../tracing/errors.js');
require('../tracing/spanstatus.js');
const trace = require('../tracing/trace.js');

/**
 * key: bucketKey
 * value: [exportKey, MetricSummary]
 */

let SPAN_METRIC_SUMMARY;

function getMetricStorageForSpan(span) {
  return SPAN_METRIC_SUMMARY ? SPAN_METRIC_SUMMARY.get(span) : undefined;
}

/**
 * Fetches the metric summary if it exists for the passed span
 */
function getMetricSummaryJsonForSpan(span) {
  const storage = getMetricStorageForSpan(span);

  if (!storage) {
    return undefined;
  }
  const output = {};

  for (const [, [exportKey, summary]] of storage) {
    if (!output[exportKey]) {
      output[exportKey] = [];
    }

    output[exportKey].push(utils.dropUndefinedKeys(summary));
  }

  return output;
}

/**
 * Updates the metric summary on the currently active span
 */
function updateMetricSummaryOnActiveSpan(
  metricType,
  sanitizedName,
  value,
  unit,
  tags,
  bucketKey,
) {
  const span = trace.getActiveSpan();
  if (span) {
    const storage = getMetricStorageForSpan(span) || new Map();

    const exportKey = `${metricType}:${sanitizedName}@${unit}`;
    const bucketItem = storage.get(bucketKey);

    if (bucketItem) {
      const [, summary] = bucketItem;
      storage.set(bucketKey, [
        exportKey,
        {
          min: Math.min(summary.min, value),
          max: Math.max(summary.max, value),
          count: (summary.count += 1),
          sum: (summary.sum += value),
          tags: summary.tags,
        },
      ]);
    } else {
      storage.set(bucketKey, [
        exportKey,
        {
          min: value,
          max: value,
          count: 1,
          sum: value,
          tags,
        },
      ]);
    }

    if (!SPAN_METRIC_SUMMARY) {
      SPAN_METRIC_SUMMARY = new WeakMap();
    }

    SPAN_METRIC_SUMMARY.set(span, storage);
  }
}

exports.getMetricSummaryJsonForSpan = getMetricSummaryJsonForSpan;
exports.updateMetricSummaryOnActiveSpan = updateMetricSummaryOnActiveSpan;
//# sourceMappingURL=metric-summary.js.map
