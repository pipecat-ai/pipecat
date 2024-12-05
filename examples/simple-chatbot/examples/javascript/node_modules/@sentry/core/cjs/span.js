Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');

/**
 * Create envelope from Span item.
 */
function createSpanEnvelope(spans, dsn) {
  const headers = {
    sent_at: new Date().toISOString(),
  };

  if (dsn) {
    headers.dsn = utils.dsnToString(dsn);
  }

  const items = spans.map(createSpanItem);
  return utils.createEnvelope(headers, items);
}

function createSpanItem(span) {
  const spanHeaders = {
    type: 'span',
  };
  return [spanHeaders, span];
}

exports.createSpanEnvelope = createSpanEnvelope;
//# sourceMappingURL=span.js.map
