Object.defineProperty(exports, '__esModule', { value: true });

const core = require('@sentry/core');
const utils = require('@sentry/utils');
const fetch = require('../common/fetch.js');
const instrument = require('./instrument.js');
const types = require('./types.js');

/* eslint-disable max-lines */

const DEFAULT_TRACE_PROPAGATION_TARGETS = ['localhost', /^\/(?!\/)/];

/** Options for Request Instrumentation */

const defaultRequestInstrumentationOptions = {
  traceFetch: true,
  traceXHR: true,
  enableHTTPTimings: true,
  // TODO (v8): Remove this property
  tracingOrigins: DEFAULT_TRACE_PROPAGATION_TARGETS,
  tracePropagationTargets: DEFAULT_TRACE_PROPAGATION_TARGETS,
};

/** Registers span creators for xhr and fetch requests  */
function instrumentOutgoingRequests(_options) {
  const {
    traceFetch,
    traceXHR,
    // eslint-disable-next-line deprecation/deprecation
    tracePropagationTargets,
    // eslint-disable-next-line deprecation/deprecation
    tracingOrigins,
    shouldCreateSpanForRequest,
    enableHTTPTimings,
  } = {
    traceFetch: defaultRequestInstrumentationOptions.traceFetch,
    traceXHR: defaultRequestInstrumentationOptions.traceXHR,
    ..._options,
  };

  const shouldCreateSpan =
    typeof shouldCreateSpanForRequest === 'function' ? shouldCreateSpanForRequest : (_) => true;

  // TODO(v8) Remove tracingOrigins here
  // The only reason we're passing it in here is because this instrumentOutgoingRequests function is publicly exported
  // and we don't want to break the API. We can remove it in v8.
  const shouldAttachHeadersWithTargets = (url) =>
    shouldAttachHeaders(url, tracePropagationTargets || tracingOrigins);

  const spans = {};

  if (traceFetch) {
    utils.addFetchInstrumentationHandler(handlerData => {
      const createdSpan = fetch.instrumentFetchRequest(handlerData, shouldCreateSpan, shouldAttachHeadersWithTargets, spans);
      // We cannot use `window.location` in the generic fetch instrumentation,
      // but we need it for reliable `server.address` attribute.
      // so we extend this in here
      if (createdSpan) {
        const fullUrl = getFullURL(handlerData.fetchData.url);
        const host = fullUrl ? utils.parseUrl(fullUrl).host : undefined;
        createdSpan.setAttributes({
          'http.url': fullUrl,
          'server.address': host,
        });
      }

      if (enableHTTPTimings && createdSpan) {
        addHTTPTimings(createdSpan);
      }
    });
  }

  if (traceXHR) {
    utils.addXhrInstrumentationHandler(handlerData => {
      const createdSpan = xhrCallback(handlerData, shouldCreateSpan, shouldAttachHeadersWithTargets, spans);
      if (enableHTTPTimings && createdSpan) {
        addHTTPTimings(createdSpan);
      }
    });
  }
}

function isPerformanceResourceTiming(entry) {
  return (
    entry.entryType === 'resource' &&
    'initiatorType' in entry &&
    typeof (entry ).nextHopProtocol === 'string' &&
    (entry.initiatorType === 'fetch' || entry.initiatorType === 'xmlhttprequest')
  );
}

/**
 * Creates a temporary observer to listen to the next fetch/xhr resourcing timings,
 * so that when timings hit their per-browser limit they don't need to be removed.
 *
 * @param span A span that has yet to be finished, must contain `url` on data.
 */
function addHTTPTimings(span) {
  const { url } = core.spanToJSON(span).data || {};

  if (!url || typeof url !== 'string') {
    return;
  }

  const cleanup = instrument.addPerformanceInstrumentationHandler('resource', ({ entries }) => {
    entries.forEach(entry => {
      if (isPerformanceResourceTiming(entry) && entry.name.endsWith(url)) {
        const spanData = resourceTimingEntryToSpanData(entry);
        spanData.forEach(data => span.setAttribute(...data));
        // In the next tick, clean this handler up
        // We have to wait here because otherwise this cleans itself up before it is fully done
        setTimeout(cleanup);
      }
    });
  });
}

/**
 * Converts ALPN protocol ids to name and version.
 *
 * (https://www.iana.org/assignments/tls-extensiontype-values/tls-extensiontype-values.xhtml#alpn-protocol-ids)
 * @param nextHopProtocol PerformanceResourceTiming.nextHopProtocol
 */
function extractNetworkProtocol(nextHopProtocol) {
  let name = 'unknown';
  let version = 'unknown';
  let _name = '';
  for (const char of nextHopProtocol) {
    // http/1.1 etc.
    if (char === '/') {
      [name, version] = nextHopProtocol.split('/');
      break;
    }
    // h2, h3 etc.
    if (!isNaN(Number(char))) {
      name = _name === 'h' ? 'http' : _name;
      version = nextHopProtocol.split(_name)[1];
      break;
    }
    _name += char;
  }
  if (_name === nextHopProtocol) {
    // webrtc, ftp, etc.
    name = _name;
  }
  return { name, version };
}

function getAbsoluteTime(time = 0) {
  return ((utils.browserPerformanceTimeOrigin || performance.timeOrigin) + time) / 1000;
}

function resourceTimingEntryToSpanData(resourceTiming) {
  const { name, version } = extractNetworkProtocol(resourceTiming.nextHopProtocol);

  const timingSpanData = [];

  timingSpanData.push(['network.protocol.version', version], ['network.protocol.name', name]);

  if (!utils.browserPerformanceTimeOrigin) {
    return timingSpanData;
  }
  return [
    ...timingSpanData,
    ['http.request.redirect_start', getAbsoluteTime(resourceTiming.redirectStart)],
    ['http.request.fetch_start', getAbsoluteTime(resourceTiming.fetchStart)],
    ['http.request.domain_lookup_start', getAbsoluteTime(resourceTiming.domainLookupStart)],
    ['http.request.domain_lookup_end', getAbsoluteTime(resourceTiming.domainLookupEnd)],
    ['http.request.connect_start', getAbsoluteTime(resourceTiming.connectStart)],
    ['http.request.secure_connection_start', getAbsoluteTime(resourceTiming.secureConnectionStart)],
    ['http.request.connection_end', getAbsoluteTime(resourceTiming.connectEnd)],
    ['http.request.request_start', getAbsoluteTime(resourceTiming.requestStart)],
    ['http.request.response_start', getAbsoluteTime(resourceTiming.responseStart)],
    ['http.request.response_end', getAbsoluteTime(resourceTiming.responseEnd)],
  ];
}

/**
 * A function that determines whether to attach tracing headers to a request.
 * This was extracted from `instrumentOutgoingRequests` to make it easier to test shouldAttachHeaders.
 * We only export this fuction for testing purposes.
 */
function shouldAttachHeaders(url, tracePropagationTargets) {
  return utils.stringMatchesSomePattern(url, tracePropagationTargets || DEFAULT_TRACE_PROPAGATION_TARGETS);
}

/**
 * Create and track xhr request spans
 *
 * @returns Span if a span was created, otherwise void.
 */
// eslint-disable-next-line complexity
function xhrCallback(
  handlerData,
  shouldCreateSpan,
  shouldAttachHeaders,
  spans,
) {
  const xhr = handlerData.xhr;
  const sentryXhrData = xhr && xhr[utils.SENTRY_XHR_DATA_KEY];

  if (!core.hasTracingEnabled() || !xhr || xhr.__sentry_own_request__ || !sentryXhrData) {
    return undefined;
  }

  const shouldCreateSpanResult = shouldCreateSpan(sentryXhrData.url);

  // check first if the request has finished and is tracked by an existing span which should now end
  if (handlerData.endTimestamp && shouldCreateSpanResult) {
    const spanId = xhr.__sentry_xhr_span_id__;
    if (!spanId) return;

    const span = spans[spanId];
    if (span && sentryXhrData.status_code !== undefined) {
      core.setHttpStatus(span, sentryXhrData.status_code);
      span.end();

      // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
      delete spans[spanId];
    }
    return undefined;
  }

  const scope = core.getCurrentScope();
  const isolationScope = core.getIsolationScope();

  const fullUrl = getFullURL(sentryXhrData.url);
  const host = fullUrl ? utils.parseUrl(fullUrl).host : undefined;

  const span = shouldCreateSpanResult
    ? core.startInactiveSpan({
        name: `${sentryXhrData.method} ${sentryXhrData.url}`,
        onlyIfParent: true,
        attributes: {
          type: 'xhr',
          'http.method': sentryXhrData.method,
          'http.url': fullUrl,
          url: sentryXhrData.url,
          'server.address': host,
          [core.SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN]: 'auto.http.browser',
        },
        op: 'http.client',
      })
    : undefined;

  if (span) {
    xhr.__sentry_xhr_span_id__ = span.spanContext().spanId;
    spans[xhr.__sentry_xhr_span_id__] = span;
  }

  const client = core.getClient();

  if (xhr.setRequestHeader && shouldAttachHeaders(sentryXhrData.url) && client) {
    const { traceId, spanId, sampled, dsc } = {
      ...isolationScope.getPropagationContext(),
      ...scope.getPropagationContext(),
    };

    const sentryTraceHeader = span ? core.spanToTraceHeader(span) : utils.generateSentryTraceHeader(traceId, spanId, sampled);

    const sentryBaggageHeader = utils.dynamicSamplingContextToSentryBaggageHeader(
      dsc ||
        (span ? core.getDynamicSamplingContextFromSpan(span) : core.getDynamicSamplingContextFromClient(traceId, client, scope)),
    );

    setHeaderOnXhr(xhr, sentryTraceHeader, sentryBaggageHeader);
  }

  return span;
}

function setHeaderOnXhr(
  xhr,
  sentryTraceHeader,
  sentryBaggageHeader,
) {
  try {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    xhr.setRequestHeader('sentry-trace', sentryTraceHeader);
    if (sentryBaggageHeader) {
      // From MDN: "If this method is called several times with the same header, the values are merged into one single request header."
      // We can therefore simply set a baggage header without checking what was there before
      // https://developer.mozilla.org/en-US/docs/Web/API/XMLHttpRequest/setRequestHeader
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      xhr.setRequestHeader(utils.BAGGAGE_HEADER_NAME, sentryBaggageHeader);
    }
  } catch (_) {
    // Error: InvalidStateError: Failed to execute 'setRequestHeader' on 'XMLHttpRequest': The object's state must be OPENED.
  }
}

function getFullURL(url) {
  try {
    // By adding a base URL to new URL(), this will also work for relative urls
    // If `url` is a full URL, the base URL is ignored anyhow
    const parsed = new URL(url, types.WINDOW.location.origin);
    return parsed.href;
  } catch (e) {
    return undefined;
  }
}

exports.DEFAULT_TRACE_PROPAGATION_TARGETS = DEFAULT_TRACE_PROPAGATION_TARGETS;
exports.defaultRequestInstrumentationOptions = defaultRequestInstrumentationOptions;
exports.extractNetworkProtocol = extractNetworkProtocol;
exports.instrumentOutgoingRequests = instrumentOutgoingRequests;
exports.shouldAttachHeaders = shouldAttachHeaders;
exports.xhrCallback = xhrCallback;
//# sourceMappingURL=request.js.map
