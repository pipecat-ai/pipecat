import { dsnToString, createEnvelope } from '@sentry/utils';

/**
 * Create envelope from Span item.
 */
function createSpanEnvelope(spans, dsn) {
  const headers = {
    sent_at: new Date().toISOString(),
  };

  if (dsn) {
    headers.dsn = dsnToString(dsn);
  }

  const items = spans.map(createSpanItem);
  return createEnvelope(headers, items);
}

function createSpanItem(span) {
  const spanHeaders = {
    type: 'span',
  };
  return [spanHeaders, span];
}

export { createSpanEnvelope };
//# sourceMappingURL=span.js.map
