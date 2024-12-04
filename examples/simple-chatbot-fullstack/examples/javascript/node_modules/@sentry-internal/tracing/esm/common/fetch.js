import { hasTracingEnabled, getCurrentScope, getClient, startInactiveSpan, SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN, getIsolationScope, spanToTraceHeader, getDynamicSamplingContextFromSpan, getDynamicSamplingContextFromClient, setHttpStatus } from '@sentry/core';
import { parseUrl, generateSentryTraceHeader, dynamicSamplingContextToSentryBaggageHeader, isInstanceOf, BAGGAGE_HEADER_NAME } from '@sentry/utils';

/**
 * Create and track fetch request spans for usage in combination with `addInstrumentationHandler`.
 *
 * @returns Span if a span was created, otherwise void.
 */
function instrumentFetchRequest(
  handlerData,
  shouldCreateSpan,
  shouldAttachHeaders,
  spans,
  spanOrigin = 'auto.http.browser',
) {
  if (!hasTracingEnabled() || !handlerData.fetchData) {
    return undefined;
  }

  const shouldCreateSpanResult = shouldCreateSpan(handlerData.fetchData.url);

  if (handlerData.endTimestamp && shouldCreateSpanResult) {
    const spanId = handlerData.fetchData.__span;
    if (!spanId) return;

    const span = spans[spanId];
    if (span) {
      endSpan(span, handlerData);
      // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
      delete spans[spanId];
    }
    return undefined;
  }

  const scope = getCurrentScope();
  const client = getClient();

  const { method, url } = handlerData.fetchData;

  const fullUrl = getFullURL(url);
  const host = fullUrl ? parseUrl(fullUrl).host : undefined;

  const span = shouldCreateSpanResult
    ? startInactiveSpan({
        name: `${method} ${url}`,
        onlyIfParent: true,
        attributes: {
          url,
          type: 'fetch',
          'http.method': method,
          'http.url': fullUrl,
          'server.address': host,
          [SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN]: spanOrigin,
        },
        op: 'http.client',
      })
    : undefined;

  if (span) {
    handlerData.fetchData.__span = span.spanContext().spanId;
    spans[span.spanContext().spanId] = span;
  }

  if (shouldAttachHeaders(handlerData.fetchData.url) && client) {
    const request = handlerData.args[0];

    // In case the user hasn't set the second argument of a fetch call we default it to `{}`.
    handlerData.args[1] = handlerData.args[1] || {};

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const options = handlerData.args[1];

    // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
    options.headers = addTracingHeadersToFetchRequest(request, client, scope, options, span);
  }

  return span;
}

/**
 * Adds sentry-trace and baggage headers to the various forms of fetch headers
 */
function addTracingHeadersToFetchRequest(
  request, // unknown is actually type Request but we can't export DOM types from this package,
  client,
  scope,
  options

,
  requestSpan,
) {
  // eslint-disable-next-line deprecation/deprecation
  const span = requestSpan || scope.getSpan();

  const isolationScope = getIsolationScope();

  const { traceId, spanId, sampled, dsc } = {
    ...isolationScope.getPropagationContext(),
    ...scope.getPropagationContext(),
  };

  const sentryTraceHeader = span ? spanToTraceHeader(span) : generateSentryTraceHeader(traceId, spanId, sampled);

  const sentryBaggageHeader = dynamicSamplingContextToSentryBaggageHeader(
    dsc ||
      (span ? getDynamicSamplingContextFromSpan(span) : getDynamicSamplingContextFromClient(traceId, client, scope)),
  );

  const headers =
    options.headers ||
    (typeof Request !== 'undefined' && isInstanceOf(request, Request) ? (request ).headers : undefined);

  if (!headers) {
    return { 'sentry-trace': sentryTraceHeader, baggage: sentryBaggageHeader };
  } else if (typeof Headers !== 'undefined' && isInstanceOf(headers, Headers)) {
    const newHeaders = new Headers(headers );

    newHeaders.append('sentry-trace', sentryTraceHeader);

    if (sentryBaggageHeader) {
      // If the same header is appended multiple times the browser will merge the values into a single request header.
      // Its therefore safe to simply push a "baggage" entry, even though there might already be another baggage header.
      newHeaders.append(BAGGAGE_HEADER_NAME, sentryBaggageHeader);
    }

    return newHeaders ;
  } else if (Array.isArray(headers)) {
    const newHeaders = [...headers, ['sentry-trace', sentryTraceHeader]];

    if (sentryBaggageHeader) {
      // If there are multiple entries with the same key, the browser will merge the values into a single request header.
      // Its therefore safe to simply push a "baggage" entry, even though there might already be another baggage header.
      newHeaders.push([BAGGAGE_HEADER_NAME, sentryBaggageHeader]);
    }

    return newHeaders ;
  } else {
    const existingBaggageHeader = 'baggage' in headers ? headers.baggage : undefined;
    const newBaggageHeaders = [];

    if (Array.isArray(existingBaggageHeader)) {
      newBaggageHeaders.push(...existingBaggageHeader);
    } else if (existingBaggageHeader) {
      newBaggageHeaders.push(existingBaggageHeader);
    }

    if (sentryBaggageHeader) {
      newBaggageHeaders.push(sentryBaggageHeader);
    }

    return {
      ...(headers ),
      'sentry-trace': sentryTraceHeader,
      baggage: newBaggageHeaders.length > 0 ? newBaggageHeaders.join(',') : undefined,
    };
  }
}

function getFullURL(url) {
  try {
    const parsed = new URL(url);
    return parsed.href;
  } catch (e) {
    return undefined;
  }
}

function endSpan(span, handlerData) {
  if (handlerData.response) {
    setHttpStatus(span, handlerData.response.status);

    const contentLength =
      handlerData.response && handlerData.response.headers && handlerData.response.headers.get('content-length');

    if (contentLength) {
      const contentLengthNum = parseInt(contentLength);
      if (contentLengthNum > 0) {
        span.setAttribute('http.response_content_length', contentLengthNum);
      }
    }
  } else if (handlerData.error) {
    span.setStatus('internal_error');
  }
  span.end();
}

export { addTracingHeadersToFetchRequest, instrumentFetchRequest };
//# sourceMappingURL=fetch.js.map
