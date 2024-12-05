Object.defineProperty(exports, '__esModule', { value: true });

const core = require('@sentry/core');
const utils = require('@sentry/utils');
const debugBuild = require('../../common/debug-build.js');
const instrument = require('../instrument.js');
const types = require('../types.js');
const getVisibilityWatcher = require('../web-vitals/lib/getVisibilityWatcher.js');
const utils$1 = require('./utils.js');
const getNavigationEntry = require('../web-vitals/lib/getNavigationEntry.js');

const MAX_INT_AS_BYTES = 2147483647;

/**
 * Converts from milliseconds to seconds
 * @param time time in ms
 */
function msToSec(time) {
  return time / 1000;
}

function getBrowserPerformanceAPI() {
  // @ts-expect-error we want to make sure all of these are available, even if TS is sure they are
  return types.WINDOW && types.WINDOW.addEventListener && types.WINDOW.performance;
}

let _performanceCursor = 0;

let _measurements = {};
let _lcpEntry;
let _clsEntry;

/**
 * Start tracking web vitals.
 * The callback returned by this function can be used to stop tracking & ensure all measurements are final & captured.
 *
 * @returns A function that forces web vitals collection
 */
function startTrackingWebVitals() {
  const performance = getBrowserPerformanceAPI();
  if (performance && utils.browserPerformanceTimeOrigin) {
    // @ts-expect-error we want to make sure all of these are available, even if TS is sure they are
    if (performance.mark) {
      types.WINDOW.performance.mark('sentry-tracing-init');
    }
    const fidCallback = _trackFID();
    const clsCallback = _trackCLS();
    const lcpCallback = _trackLCP();
    const ttfbCallback = _trackTtfb();

    return () => {
      fidCallback();
      clsCallback();
      lcpCallback();
      ttfbCallback();
    };
  }

  return () => undefined;
}

/**
 * Start tracking long tasks.
 */
function startTrackingLongTasks() {
  instrument.addPerformanceInstrumentationHandler('longtask', ({ entries }) => {
    for (const entry of entries) {
      // eslint-disable-next-line deprecation/deprecation
      const transaction = core.getActiveTransaction() ;
      if (!transaction) {
        return;
      }
      const startTime = msToSec((utils.browserPerformanceTimeOrigin ) + entry.startTime);
      const duration = msToSec(entry.duration);

      // eslint-disable-next-line deprecation/deprecation
      transaction.startChild({
        description: 'Main UI thread blocked',
        op: 'ui.long-task',
        origin: 'auto.ui.browser.metrics',
        startTimestamp: startTime,
        endTimestamp: startTime + duration,
      });
    }
  });
}

/**
 * Start tracking interaction events.
 */
function startTrackingInteractions() {
  instrument.addPerformanceInstrumentationHandler('event', ({ entries }) => {
    for (const entry of entries) {
      // eslint-disable-next-line deprecation/deprecation
      const transaction = core.getActiveTransaction() ;
      if (!transaction) {
        return;
      }

      if (entry.name === 'click') {
        const startTime = msToSec((utils.browserPerformanceTimeOrigin ) + entry.startTime);
        const duration = msToSec(entry.duration);

        const span = {
          description: utils.htmlTreeAsString(entry.target),
          op: `ui.interaction.${entry.name}`,
          origin: 'auto.ui.browser.metrics',
          startTimestamp: startTime,
          endTimestamp: startTime + duration,
        };

        const componentName = utils.getComponentName(entry.target);
        if (componentName) {
          span.attributes = { 'ui.component_name': componentName };
        }

        // eslint-disable-next-line deprecation/deprecation
        transaction.startChild(span);
      }
    }
  });
}

/**
 * Start tracking INP webvital events.
 */
function startTrackingINP(
  interactionIdtoRouteNameMapping,
  interactionsSampleRate,
) {
  const performance = getBrowserPerformanceAPI();
  if (performance && utils.browserPerformanceTimeOrigin) {
    const inpCallback = _trackINP(interactionIdtoRouteNameMapping, interactionsSampleRate);

    return () => {
      inpCallback();
    };
  }

  return () => undefined;
}

/** Starts tracking the Cumulative Layout Shift on the current page. */
function _trackCLS() {
  return instrument.addClsInstrumentationHandler(({ metric }) => {
    const entry = metric.entries[metric.entries.length - 1];
    if (!entry) {
      return;
    }

    debugBuild.DEBUG_BUILD && utils.logger.log('[Measurements] Adding CLS');
    _measurements['cls'] = { value: metric.value, unit: '' };
    _clsEntry = entry ;
  }, true);
}

/** Starts tracking the Largest Contentful Paint on the current page. */
function _trackLCP() {
  return instrument.addLcpInstrumentationHandler(({ metric }) => {
    const entry = metric.entries[metric.entries.length - 1];
    if (!entry) {
      return;
    }

    debugBuild.DEBUG_BUILD && utils.logger.log('[Measurements] Adding LCP');
    _measurements['lcp'] = { value: metric.value, unit: 'millisecond' };
    _lcpEntry = entry ;
  }, true);
}

/** Starts tracking the First Input Delay on the current page. */
function _trackFID() {
  return instrument.addFidInstrumentationHandler(({ metric }) => {
    const entry = metric.entries[metric.entries.length - 1];
    if (!entry) {
      return;
    }

    const timeOrigin = msToSec(utils.browserPerformanceTimeOrigin );
    const startTime = msToSec(entry.startTime);
    debugBuild.DEBUG_BUILD && utils.logger.log('[Measurements] Adding FID');
    _measurements['fid'] = { value: metric.value, unit: 'millisecond' };
    _measurements['mark.fid'] = { value: timeOrigin + startTime, unit: 'second' };
  });
}

function _trackTtfb() {
  return instrument.addTtfbInstrumentationHandler(({ metric }) => {
    const entry = metric.entries[metric.entries.length - 1];
    if (!entry) {
      return;
    }

    debugBuild.DEBUG_BUILD && utils.logger.log('[Measurements] Adding TTFB');
    _measurements['ttfb'] = { value: metric.value, unit: 'millisecond' };
  });
}

const INP_ENTRY_MAP = {
  click: 'click',
  pointerdown: 'click',
  pointerup: 'click',
  mousedown: 'click',
  mouseup: 'click',
  touchstart: 'click',
  touchend: 'click',
  mouseover: 'hover',
  mouseout: 'hover',
  mouseenter: 'hover',
  mouseleave: 'hover',
  pointerover: 'hover',
  pointerout: 'hover',
  pointerenter: 'hover',
  pointerleave: 'hover',
  dragstart: 'drag',
  dragend: 'drag',
  drag: 'drag',
  dragenter: 'drag',
  dragleave: 'drag',
  dragover: 'drag',
  drop: 'drag',
  keydown: 'press',
  keyup: 'press',
  keypress: 'press',
  input: 'press',
};

/** Starts tracking the Interaction to Next Paint on the current page. */
function _trackINP(
  interactionIdToRouteNameMapping,
  interactionsSampleRate,
) {
  return instrument.addInpInstrumentationHandler(({ metric }) => {
    if (metric.value === undefined) {
      return;
    }
    const entry = metric.entries.find(
      entry => entry.duration === metric.value && INP_ENTRY_MAP[entry.name] !== undefined,
    );
    const client = core.getClient();
    if (!entry || !client) {
      return;
    }
    const interactionType = INP_ENTRY_MAP[entry.name];
    const options = client.getOptions();
    /** Build the INP span, create an envelope from the span, and then send the envelope */
    const startTime = msToSec((utils.browserPerformanceTimeOrigin ) + entry.startTime);
    const duration = msToSec(metric.value);
    const interaction =
      entry.interactionId !== undefined ? interactionIdToRouteNameMapping[entry.interactionId] : undefined;
    if (interaction === undefined) {
      return;
    }
    const { routeName, parentContext, activeTransaction, user, replayId } = interaction;
    const userDisplay = user !== undefined ? user.email || user.id || user.ip_address : undefined;
    // eslint-disable-next-line deprecation/deprecation
    const profileId = activeTransaction !== undefined ? activeTransaction.getProfileId() : undefined;
    const span = new core.Span({
      startTimestamp: startTime,
      endTimestamp: startTime + duration,
      op: `ui.interaction.${interactionType}`,
      name: utils.htmlTreeAsString(entry.target),
      attributes: {
        release: options.release,
        environment: options.environment,
        transaction: routeName,
        ...(userDisplay !== undefined && userDisplay !== '' ? { user: userDisplay } : {}),
        ...(profileId !== undefined ? { profile_id: profileId } : {}),
        ...(replayId !== undefined ? { replay_id: replayId } : {}),
      },
      exclusiveTime: metric.value,
      measurements: {
        inp: { value: metric.value, unit: 'millisecond' },
      },
    });

    /** Check to see if the span should be sampled */
    const sampleRate = getSampleRate(parentContext, options, interactionsSampleRate);

    if (!sampleRate) {
      return;
    }

    if (Math.random() < (sampleRate )) {
      const envelope = span ? core.createSpanEnvelope([span], client.getDsn()) : undefined;
      const transport = client && client.getTransport();
      if (transport && envelope) {
        transport.send(envelope).then(null, reason => {
          debugBuild.DEBUG_BUILD && utils.logger.error('Error while sending interaction:', reason);
        });
      }
      return;
    }
  });
}

/** Add performance related spans to a transaction */
function addPerformanceEntries(transaction) {
  const performance = getBrowserPerformanceAPI();
  if (!performance || !types.WINDOW.performance.getEntries || !utils.browserPerformanceTimeOrigin) {
    // Gatekeeper if performance API not available
    return;
  }

  debugBuild.DEBUG_BUILD && utils.logger.log('[Tracing] Adding & adjusting spans using Performance API');
  const timeOrigin = msToSec(utils.browserPerformanceTimeOrigin);

  const performanceEntries = performance.getEntries();

  const { op, start_timestamp: transactionStartTime } = core.spanToJSON(transaction);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  performanceEntries.slice(_performanceCursor).forEach((entry) => {
    const startTime = msToSec(entry.startTime);
    const duration = msToSec(entry.duration);

    // eslint-disable-next-line deprecation/deprecation
    if (transaction.op === 'navigation' && transactionStartTime && timeOrigin + startTime < transactionStartTime) {
      return;
    }

    switch (entry.entryType) {
      case 'navigation': {
        _addNavigationSpans(transaction, entry, timeOrigin);
        break;
      }
      case 'mark':
      case 'paint':
      case 'measure': {
        _addMeasureSpans(transaction, entry, startTime, duration, timeOrigin);

        // capture web vitals
        const firstHidden = getVisibilityWatcher.getVisibilityWatcher();
        // Only report if the page wasn't hidden prior to the web vital.
        const shouldRecord = entry.startTime < firstHidden.firstHiddenTime;

        if (entry.name === 'first-paint' && shouldRecord) {
          debugBuild.DEBUG_BUILD && utils.logger.log('[Measurements] Adding FP');
          _measurements['fp'] = { value: entry.startTime, unit: 'millisecond' };
        }
        if (entry.name === 'first-contentful-paint' && shouldRecord) {
          debugBuild.DEBUG_BUILD && utils.logger.log('[Measurements] Adding FCP');
          _measurements['fcp'] = { value: entry.startTime, unit: 'millisecond' };
        }
        break;
      }
      case 'resource': {
        _addResourceSpans(transaction, entry, entry.name , startTime, duration, timeOrigin);
        break;
      }
      // Ignore other entry types.
    }
  });

  _performanceCursor = Math.max(performanceEntries.length - 1, 0);

  _trackNavigator(transaction);

  // Measurements are only available for pageload transactions
  if (op === 'pageload') {
    _addTtfbRequestTimeToMeasurements(_measurements);

    ['fcp', 'fp', 'lcp'].forEach(name => {
      if (!_measurements[name] || !transactionStartTime || timeOrigin >= transactionStartTime) {
        return;
      }
      // The web vitals, fcp, fp, lcp, and ttfb, all measure relative to timeOrigin.
      // Unfortunately, timeOrigin is not captured within the transaction span data, so these web vitals will need
      // to be adjusted to be relative to transaction.startTimestamp.
      const oldValue = _measurements[name].value;
      const measurementTimestamp = timeOrigin + msToSec(oldValue);

      // normalizedValue should be in milliseconds
      const normalizedValue = Math.abs((measurementTimestamp - transactionStartTime) * 1000);
      const delta = normalizedValue - oldValue;

      debugBuild.DEBUG_BUILD && utils.logger.log(`[Measurements] Normalized ${name} from ${oldValue} to ${normalizedValue} (${delta})`);
      _measurements[name].value = normalizedValue;
    });

    const fidMark = _measurements['mark.fid'];
    if (fidMark && _measurements['fid']) {
      // create span for FID
      utils$1._startChild(transaction, {
        description: 'first input delay',
        endTimestamp: fidMark.value + msToSec(_measurements['fid'].value),
        op: 'ui.action',
        origin: 'auto.ui.browser.metrics',
        startTimestamp: fidMark.value,
      });

      // Delete mark.fid as we don't want it to be part of final payload
      delete _measurements['mark.fid'];
    }

    // If FCP is not recorded we should not record the cls value
    // according to the new definition of CLS.
    if (!('fcp' in _measurements)) {
      delete _measurements.cls;
    }

    Object.keys(_measurements).forEach(measurementName => {
      core.setMeasurement(measurementName, _measurements[measurementName].value, _measurements[measurementName].unit);
    });

    _tagMetricInfo(transaction);
  }

  _lcpEntry = undefined;
  _clsEntry = undefined;
  _measurements = {};
}

/** Create measure related spans */
function _addMeasureSpans(
  transaction,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  entry,
  startTime,
  duration,
  timeOrigin,
) {
  const measureStartTimestamp = timeOrigin + startTime;
  const measureEndTimestamp = measureStartTimestamp + duration;

  utils$1._startChild(transaction, {
    description: entry.name ,
    endTimestamp: measureEndTimestamp,
    op: entry.entryType ,
    origin: 'auto.resource.browser.metrics',
    startTimestamp: measureStartTimestamp,
  });

  return measureStartTimestamp;
}

/** Instrument navigation entries */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _addNavigationSpans(transaction, entry, timeOrigin) {
  ['unloadEvent', 'redirect', 'domContentLoadedEvent', 'loadEvent', 'connect'].forEach(event => {
    _addPerformanceNavigationTiming(transaction, entry, event, timeOrigin);
  });
  _addPerformanceNavigationTiming(transaction, entry, 'secureConnection', timeOrigin, 'TLS/SSL', 'connectEnd');
  _addPerformanceNavigationTiming(transaction, entry, 'fetch', timeOrigin, 'cache', 'domainLookupStart');
  _addPerformanceNavigationTiming(transaction, entry, 'domainLookup', timeOrigin, 'DNS');
  _addRequest(transaction, entry, timeOrigin);
}

/** Create performance navigation related spans */
function _addPerformanceNavigationTiming(
  transaction,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  entry,
  event,
  timeOrigin,
  description,
  eventEnd,
) {
  const end = eventEnd ? (entry[eventEnd] ) : (entry[`${event}End`] );
  const start = entry[`${event}Start`] ;
  if (!start || !end) {
    return;
  }
  utils$1._startChild(transaction, {
    op: 'browser',
    origin: 'auto.browser.browser.metrics',
    description: description || event,
    startTimestamp: timeOrigin + msToSec(start),
    endTimestamp: timeOrigin + msToSec(end),
  });
}

/** Create request and response related spans */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _addRequest(transaction, entry, timeOrigin) {
  if (entry.responseEnd) {
    // It is possible that we are collecting these metrics when the page hasn't finished loading yet, for example when the HTML slowly streams in.
    // In this case, ie. when the document request hasn't finished yet, `entry.responseEnd` will be 0.
    // In order not to produce faulty spans, where the end timestamp is before the start timestamp, we will only collect
    // these spans when the responseEnd value is available. The backend (Relay) would drop the entire transaction if it contained faulty spans.
    utils$1._startChild(transaction, {
      op: 'browser',
      origin: 'auto.browser.browser.metrics',
      description: 'request',
      startTimestamp: timeOrigin + msToSec(entry.requestStart ),
      endTimestamp: timeOrigin + msToSec(entry.responseEnd ),
    });

    utils$1._startChild(transaction, {
      op: 'browser',
      origin: 'auto.browser.browser.metrics',
      description: 'response',
      startTimestamp: timeOrigin + msToSec(entry.responseStart ),
      endTimestamp: timeOrigin + msToSec(entry.responseEnd ),
    });
  }
}

/** Create resource-related spans */
function _addResourceSpans(
  transaction,
  entry,
  resourceUrl,
  startTime,
  duration,
  timeOrigin,
) {
  // we already instrument based on fetch and xhr, so we don't need to
  // duplicate spans here.
  if (entry.initiatorType === 'xmlhttprequest' || entry.initiatorType === 'fetch') {
    return;
  }

  const parsedUrl = utils.parseUrl(resourceUrl);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data = {};
  setResourceEntrySizeData(data, entry, 'transferSize', 'http.response_transfer_size');
  setResourceEntrySizeData(data, entry, 'encodedBodySize', 'http.response_content_length');
  setResourceEntrySizeData(data, entry, 'decodedBodySize', 'http.decoded_response_content_length');

  if ('renderBlockingStatus' in entry) {
    data['resource.render_blocking_status'] = entry.renderBlockingStatus;
  }
  if (parsedUrl.protocol) {
    data['url.scheme'] = parsedUrl.protocol.split(':').pop(); // the protocol returned by parseUrl includes a :, but OTEL spec does not, so we remove it.
  }

  if (parsedUrl.host) {
    data['server.address'] = parsedUrl.host;
  }

  data['url.same_origin'] = resourceUrl.includes(types.WINDOW.location.origin);

  const startTimestamp = timeOrigin + startTime;
  const endTimestamp = startTimestamp + duration;

  utils$1._startChild(transaction, {
    description: resourceUrl.replace(types.WINDOW.location.origin, ''),
    endTimestamp,
    op: entry.initiatorType ? `resource.${entry.initiatorType}` : 'resource.other',
    origin: 'auto.resource.browser.metrics',
    startTimestamp,
    data,
  });
}

/**
 * Capture the information of the user agent.
 */
function _trackNavigator(transaction) {
  const navigator = types.WINDOW.navigator ;
  if (!navigator) {
    return;
  }

  // track network connectivity
  const connection = navigator.connection;
  if (connection) {
    if (connection.effectiveType) {
      // TODO: Can we rewrite this to an attribute?
      // eslint-disable-next-line deprecation/deprecation
      transaction.setTag('effectiveConnectionType', connection.effectiveType);
    }

    if (connection.type) {
      // TODO: Can we rewrite this to an attribute?
      // eslint-disable-next-line deprecation/deprecation
      transaction.setTag('connectionType', connection.type);
    }

    if (utils$1.isMeasurementValue(connection.rtt)) {
      _measurements['connection.rtt'] = { value: connection.rtt, unit: 'millisecond' };
    }
  }

  if (utils$1.isMeasurementValue(navigator.deviceMemory)) {
    // TODO: Can we rewrite this to an attribute?
    // eslint-disable-next-line deprecation/deprecation
    transaction.setTag('deviceMemory', `${navigator.deviceMemory} GB`);
  }

  if (utils$1.isMeasurementValue(navigator.hardwareConcurrency)) {
    // TODO: Can we rewrite this to an attribute?
    // eslint-disable-next-line deprecation/deprecation
    transaction.setTag('hardwareConcurrency', String(navigator.hardwareConcurrency));
  }
}

/** Add LCP / CLS data to transaction to allow debugging */
function _tagMetricInfo(transaction) {
  if (_lcpEntry) {
    debugBuild.DEBUG_BUILD && utils.logger.log('[Measurements] Adding LCP Data');

    // Capture Properties of the LCP element that contributes to the LCP.

    if (_lcpEntry.element) {
      // TODO: Can we rewrite this to an attribute?
      // eslint-disable-next-line deprecation/deprecation
      transaction.setTag('lcp.element', utils.htmlTreeAsString(_lcpEntry.element));
    }

    if (_lcpEntry.id) {
      // TODO: Can we rewrite this to an attribute?
      // eslint-disable-next-line deprecation/deprecation
      transaction.setTag('lcp.id', _lcpEntry.id);
    }

    if (_lcpEntry.url) {
      // Trim URL to the first 200 characters.
      // TODO: Can we rewrite this to an attribute?
      // eslint-disable-next-line deprecation/deprecation
      transaction.setTag('lcp.url', _lcpEntry.url.trim().slice(0, 200));
    }

    // TODO: Can we rewrite this to an attribute?
    // eslint-disable-next-line deprecation/deprecation
    transaction.setTag('lcp.size', _lcpEntry.size);
  }

  // See: https://developer.mozilla.org/en-US/docs/Web/API/LayoutShift
  if (_clsEntry && _clsEntry.sources) {
    debugBuild.DEBUG_BUILD && utils.logger.log('[Measurements] Adding CLS Data');
    _clsEntry.sources.forEach((source, index) =>
      // TODO: Can we rewrite this to an attribute?
      // eslint-disable-next-line deprecation/deprecation
      transaction.setTag(`cls.source.${index + 1}`, utils.htmlTreeAsString(source.node)),
    );
  }
}

function setResourceEntrySizeData(
  data,
  entry,
  key,
  dataKey,
) {
  const entryVal = entry[key];
  if (entryVal != null && entryVal < MAX_INT_AS_BYTES) {
    data[dataKey] = entryVal;
  }
}

/**
 * Add ttfb request time information to measurements.
 *
 * ttfb information is added via vendored web vitals library.
 */
function _addTtfbRequestTimeToMeasurements(_measurements) {
  const navEntry = getNavigationEntry.getNavigationEntry();
  if (!navEntry) {
    return;
  }

  const { responseStart, requestStart } = navEntry;

  if (requestStart <= responseStart) {
    debugBuild.DEBUG_BUILD && utils.logger.log('[Measurements] Adding TTFB Request Time');
    _measurements['ttfb.requestTime'] = {
      value: responseStart - requestStart,
      unit: 'millisecond',
    };
  }
}

/** Taken from @sentry/core sampling.ts */
function getSampleRate(
  transactionContext,
  options,
  interactionsSampleRate,
) {
  if (!core.hasTracingEnabled(options)) {
    return false;
  }
  let sampleRate;
  if (transactionContext !== undefined && typeof options.tracesSampler === 'function') {
    sampleRate = options.tracesSampler({
      transactionContext,
      name: transactionContext.name,
      parentSampled: transactionContext.parentSampled,
      attributes: {
        // eslint-disable-next-line deprecation/deprecation
        ...transactionContext.data,
        ...transactionContext.attributes,
      },
      location: types.WINDOW.location,
    });
  } else if (transactionContext !== undefined && transactionContext.sampled !== undefined) {
    sampleRate = transactionContext.sampled;
  } else if (typeof options.tracesSampleRate !== 'undefined') {
    sampleRate = options.tracesSampleRate;
  } else {
    sampleRate = 1;
  }
  if (!core.isValidSampleRate(sampleRate)) {
    debugBuild.DEBUG_BUILD && utils.logger.warn('[Tracing] Discarding interaction span because of invalid sample rate.');
    return false;
  }
  if (sampleRate === true) {
    return interactionsSampleRate;
  } else if (sampleRate === false) {
    return 0;
  }
  return sampleRate * interactionsSampleRate;
}

exports._addMeasureSpans = _addMeasureSpans;
exports._addResourceSpans = _addResourceSpans;
exports.addPerformanceEntries = addPerformanceEntries;
exports.startTrackingINP = startTrackingINP;
exports.startTrackingInteractions = startTrackingInteractions;
exports.startTrackingLongTasks = startTrackingLongTasks;
exports.startTrackingWebVitals = startTrackingWebVitals;
//# sourceMappingURL=index.js.map
