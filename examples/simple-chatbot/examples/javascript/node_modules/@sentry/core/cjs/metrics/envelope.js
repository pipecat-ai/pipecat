Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const utils$1 = require('./utils.js');

/**
 * Create envelope from a metric aggregate.
 */
function createMetricEnvelope(
  metricBucketItems,
  dsn,
  metadata,
  tunnel,
) {
  const headers = {
    sent_at: new Date().toISOString(),
  };

  if (metadata && metadata.sdk) {
    headers.sdk = {
      name: metadata.sdk.name,
      version: metadata.sdk.version,
    };
  }

  if (!!tunnel && dsn) {
    headers.dsn = utils.dsnToString(dsn);
  }

  const item = createMetricEnvelopeItem(metricBucketItems);
  return utils.createEnvelope(headers, [item]);
}

function createMetricEnvelopeItem(metricBucketItems) {
  const payload = utils$1.serializeMetricBuckets(metricBucketItems);
  const metricHeaders = {
    type: 'statsd',
    length: payload.length,
  };
  return [metricHeaders, payload];
}

exports.createMetricEnvelope = createMetricEnvelope;
//# sourceMappingURL=envelope.js.map
