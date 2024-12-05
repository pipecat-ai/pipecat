import { TRACING_DEFAULTS, addTracingExtensions, spanToJSON, SEMANTIC_ATTRIBUTE_SENTRY_SOURCE, getActiveSpan, getCurrentHub, startIdleTransaction, getActiveTransaction, getClient, getCurrentScope } from '@sentry/core';
import { logger, browserPerformanceTimeOrigin, addHistoryInstrumentationHandler, propagationContextFromHeaders, getDomElement } from '@sentry/utils';
import { DEBUG_BUILD } from '../common/debug-build.js';
import { registerBackgroundTabDetection } from './backgroundtab.js';
import { addPerformanceInstrumentationHandler } from './instrument.js';
import { startTrackingWebVitals, startTrackingINP, startTrackingLongTasks, startTrackingInteractions, addPerformanceEntries } from './metrics/index.js';
import { defaultRequestInstrumentationOptions, instrumentOutgoingRequests } from './request.js';
import { WINDOW } from './types.js';

const BROWSER_TRACING_INTEGRATION_ID = 'BrowserTracing';

/** Options for Browser Tracing integration */

const DEFAULT_BROWSER_TRACING_OPTIONS = {
  ...TRACING_DEFAULTS,
  instrumentNavigation: true,
  instrumentPageLoad: true,
  markBackgroundSpan: true,
  enableLongTask: true,
  enableInp: false,
  interactionsSampleRate: 1,
  _experiments: {},
  ...defaultRequestInstrumentationOptions,
};

/**
 * The Browser Tracing integration automatically instruments browser pageload/navigation
 * actions as transactions, and captures requests, metrics and errors as spans.
 *
 * The integration can be configured with a variety of options, and can be extended to use
 * any routing library. This integration uses {@see IdleTransaction} to create transactions.
 *
 * We explicitly export the proper type here, as this has to be extended in some cases.
 */
const browserTracingIntegration = ((_options = {}) => {
  const _hasSetTracePropagationTargets = DEBUG_BUILD
    ? !!(
        // eslint-disable-next-line deprecation/deprecation
        (_options.tracePropagationTargets || _options.tracingOrigins)
      )
    : false;

  addTracingExtensions();

  // TODO (v8): remove this block after tracingOrigins is removed
  // Set tracePropagationTargets to tracingOrigins if specified by the user
  // In case both are specified, tracePropagationTargets takes precedence
  // eslint-disable-next-line deprecation/deprecation
  if (!_options.tracePropagationTargets && _options.tracingOrigins) {
    // eslint-disable-next-line deprecation/deprecation
    _options.tracePropagationTargets = _options.tracingOrigins;
  }

  const options = {
    ...DEFAULT_BROWSER_TRACING_OPTIONS,
    ..._options,
  };

  const _collectWebVitals = startTrackingWebVitals();

  /** Stores a mapping of interactionIds from PerformanceEventTimings to the origin interaction path */
  const interactionIdToRouteNameMapping = {};
  if (options.enableInp) {
    startTrackingINP(interactionIdToRouteNameMapping, options.interactionsSampleRate);
  }

  if (options.enableLongTask) {
    startTrackingLongTasks();
  }
  if (options._experiments.enableInteractions) {
    startTrackingInteractions();
  }

  const latestRoute

 = {
    name: undefined,
    context: undefined,
  };

  /** Create routing idle transaction. */
  function _createRouteTransaction(context) {
    // eslint-disable-next-line deprecation/deprecation
    const hub = getCurrentHub();

    const { beforeStartSpan, idleTimeout, finalTimeout, heartbeatInterval } = options;

    const isPageloadTransaction = context.op === 'pageload';

    let expandedContext;
    if (isPageloadTransaction) {
      const sentryTrace = isPageloadTransaction ? getMetaContent('sentry-trace') : '';
      const baggage = isPageloadTransaction ? getMetaContent('baggage') : undefined;
      const { traceId, dsc, parentSpanId, sampled } = propagationContextFromHeaders(sentryTrace, baggage);
      expandedContext = {
        traceId,
        parentSpanId,
        parentSampled: sampled,
        ...context,
        metadata: {
          // eslint-disable-next-line deprecation/deprecation
          ...context.metadata,
          dynamicSamplingContext: dsc,
        },
        trimEnd: true,
      };
    } else {
      expandedContext = {
        trimEnd: true,
        ...context,
      };
    }

    const finalContext = beforeStartSpan ? beforeStartSpan(expandedContext) : expandedContext;

    // If `beforeStartSpan` set a custom name, record that fact
    // eslint-disable-next-line deprecation/deprecation
    finalContext.metadata =
      finalContext.name !== expandedContext.name
        ? // eslint-disable-next-line deprecation/deprecation
          { ...finalContext.metadata, source: 'custom' }
        : // eslint-disable-next-line deprecation/deprecation
          finalContext.metadata;

    latestRoute.name = finalContext.name;
    latestRoute.context = finalContext;

    if (finalContext.sampled === false) {
      DEBUG_BUILD && logger.log(`[Tracing] Will not send ${finalContext.op} transaction because of beforeNavigate.`);
    }

    DEBUG_BUILD && logger.log(`[Tracing] Starting ${finalContext.op} transaction on scope`);

    const { location } = WINDOW;

    const idleTransaction = startIdleTransaction(
      hub,
      finalContext,
      idleTimeout,
      finalTimeout,
      true,
      { location }, // for use in the tracesSampler
      heartbeatInterval,
      isPageloadTransaction, // should wait for finish signal if it's a pageload transaction
    );

    if (isPageloadTransaction && WINDOW.document) {
      WINDOW.document.addEventListener('readystatechange', () => {
        if (['interactive', 'complete'].includes(WINDOW.document.readyState)) {
          idleTransaction.sendAutoFinishSignal();
        }
      });

      if (['interactive', 'complete'].includes(WINDOW.document.readyState)) {
        idleTransaction.sendAutoFinishSignal();
      }
    }

    idleTransaction.registerBeforeFinishCallback(transaction => {
      _collectWebVitals();
      addPerformanceEntries(transaction);
    });

    return idleTransaction ;
  }

  return {
    name: BROWSER_TRACING_INTEGRATION_ID,
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    setupOnce: () => {},
    afterAllSetup(client) {
      const clientOptions = client.getOptions();

      const { markBackgroundSpan, traceFetch, traceXHR, shouldCreateSpanForRequest, enableHTTPTimings, _experiments } =
        options;

      const clientOptionsTracePropagationTargets = clientOptions && clientOptions.tracePropagationTargets;
      // There are three ways to configure tracePropagationTargets:
      // 1. via top level client option `tracePropagationTargets`
      // 2. via BrowserTracing option `tracePropagationTargets`
      // 3. via BrowserTracing option `tracingOrigins` (deprecated)
      //
      // To avoid confusion, favour top level client option `tracePropagationTargets`, and fallback to
      // BrowserTracing option `tracePropagationTargets` and then `tracingOrigins` (deprecated).
      // This is done as it minimizes bundle size (we don't have to have undefined checks).
      //
      // If both 1 and either one of 2 or 3 are set (from above), we log out a warning.
      // eslint-disable-next-line deprecation/deprecation
      const tracePropagationTargets = clientOptionsTracePropagationTargets || options.tracePropagationTargets;
      if (DEBUG_BUILD && _hasSetTracePropagationTargets && clientOptionsTracePropagationTargets) {
        logger.warn(
          '[Tracing] The `tracePropagationTargets` option was set in the BrowserTracing integration and top level `Sentry.init`. The top level `Sentry.init` value is being used.',
        );
      }

      let activeSpan;
      let startingUrl = WINDOW.location && WINDOW.location.href;

      if (client.on) {
        client.on('startNavigationSpan', (context) => {
          if (activeSpan) {
            DEBUG_BUILD && logger.log(`[Tracing] Finishing current transaction with op: ${spanToJSON(activeSpan).op}`);
            // If there's an open transaction on the scope, we need to finish it before creating an new one.
            activeSpan.end();
          }
          activeSpan = _createRouteTransaction({
            op: 'navigation',
            ...context,
          });
        });

        client.on('startPageLoadSpan', (context) => {
          if (activeSpan) {
            DEBUG_BUILD && logger.log(`[Tracing] Finishing current transaction with op: ${spanToJSON(activeSpan).op}`);
            // If there's an open transaction on the scope, we need to finish it before creating an new one.
            activeSpan.end();
          }
          activeSpan = _createRouteTransaction({
            op: 'pageload',
            ...context,
          });
        });
      }

      if (options.instrumentPageLoad && client.emit && WINDOW.location) {
        const context = {
          name: WINDOW.location.pathname,
          // pageload should always start at timeOrigin (and needs to be in s, not ms)
          startTimestamp: browserPerformanceTimeOrigin ? browserPerformanceTimeOrigin / 1000 : undefined,
          origin: 'auto.pageload.browser',
          attributes: {
            [SEMANTIC_ATTRIBUTE_SENTRY_SOURCE]: 'url',
          },
        };
        startBrowserTracingPageLoadSpan(client, context);
      }

      if (options.instrumentNavigation && client.emit && WINDOW.location) {
        addHistoryInstrumentationHandler(({ to, from }) => {
          /**
           * This early return is there to account for some cases where a navigation transaction starts right after
           * long-running pageload. We make sure that if `from` is undefined and a valid `startingURL` exists, we don't
           * create an uneccessary navigation transaction.
           *
           * This was hard to duplicate, but this behavior stopped as soon as this fix was applied. This issue might also
           * only be caused in certain development environments where the usage of a hot module reloader is causing
           * errors.
           */
          if (from === undefined && startingUrl && startingUrl.indexOf(to) !== -1) {
            startingUrl = undefined;
            return;
          }

          if (from !== to) {
            startingUrl = undefined;
            const context = {
              name: WINDOW.location.pathname,
              origin: 'auto.navigation.browser',
              attributes: {
                [SEMANTIC_ATTRIBUTE_SENTRY_SOURCE]: 'url',
              },
            };

            startBrowserTracingNavigationSpan(client, context);
          }
        });
      }

      if (markBackgroundSpan) {
        registerBackgroundTabDetection();
      }

      if (_experiments.enableInteractions) {
        registerInteractionListener(options, latestRoute);
      }

      if (options.enableInp) {
        registerInpInteractionListener(interactionIdToRouteNameMapping, latestRoute);
      }

      instrumentOutgoingRequests({
        traceFetch,
        traceXHR,
        tracePropagationTargets,
        shouldCreateSpanForRequest,
        enableHTTPTimings,
      });
    },
    // TODO v8: Remove this again
    // This is private API that we use to fix converted BrowserTracing integrations in Next.js & SvelteKit
    options,
  };
}) ;

/**
 * Manually start a page load span.
 * This will only do something if the BrowserTracing integration has been setup.
 */
function startBrowserTracingPageLoadSpan(client, spanOptions) {
  if (!client.emit) {
    return;
  }

  client.emit('startPageLoadSpan', spanOptions);

  const span = getActiveSpan();
  const op = span && spanToJSON(span).op;
  return op === 'pageload' ? span : undefined;
}

/**
 * Manually start a navigation span.
 * This will only do something if the BrowserTracing integration has been setup.
 */
function startBrowserTracingNavigationSpan(client, spanOptions) {
  if (!client.emit) {
    return;
  }

  client.emit('startNavigationSpan', spanOptions);

  const span = getActiveSpan();
  const op = span && spanToJSON(span).op;
  return op === 'navigation' ? span : undefined;
}

/** Returns the value of a meta tag */
function getMetaContent(metaName) {
  // Can't specify generic to `getDomElement` because tracing can be used
  // in a variety of environments, have to disable `no-unsafe-member-access`
  // as a result.
  const metaTag = getDomElement(`meta[name=${metaName}]`);
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  return metaTag ? metaTag.getAttribute('content') : undefined;
}

/** Start listener for interaction transactions */
function registerInteractionListener(
  options,
  latestRoute

,
) {
  let inflightInteractionTransaction;
  const registerInteractionTransaction = () => {
    const { idleTimeout, finalTimeout, heartbeatInterval } = options;
    const op = 'ui.action.click';

    // eslint-disable-next-line deprecation/deprecation
    const currentTransaction = getActiveTransaction();
    if (currentTransaction && currentTransaction.op && ['navigation', 'pageload'].includes(currentTransaction.op)) {
      DEBUG_BUILD &&
        logger.warn(
          `[Tracing] Did not create ${op} transaction because a pageload or navigation transaction is in progress.`,
        );
      return undefined;
    }

    if (inflightInteractionTransaction) {
      inflightInteractionTransaction.setFinishReason('interactionInterrupted');
      inflightInteractionTransaction.end();
      inflightInteractionTransaction = undefined;
    }

    if (!latestRoute.name) {
      DEBUG_BUILD && logger.warn(`[Tracing] Did not create ${op} transaction because _latestRouteName is missing.`);
      return undefined;
    }

    const { location } = WINDOW;

    const context = {
      name: latestRoute.name,
      op,
      trimEnd: true,
      data: {
        [SEMANTIC_ATTRIBUTE_SENTRY_SOURCE]: latestRoute.context ? getSource(latestRoute.context) : 'url',
      },
    };

    inflightInteractionTransaction = startIdleTransaction(
      // eslint-disable-next-line deprecation/deprecation
      getCurrentHub(),
      context,
      idleTimeout,
      finalTimeout,
      true,
      { location }, // for use in the tracesSampler
      heartbeatInterval,
    );
  };

  ['click'].forEach(type => {
    if (WINDOW.document) {
      addEventListener(type, registerInteractionTransaction, { once: false, capture: true });
    }
  });
}

function isPerformanceEventTiming(entry) {
  return 'duration' in entry;
}

/** We store up to 10 interaction candidates max to cap memory usage. This is the same cap as getINP from web-vitals */
const MAX_INTERACTIONS = 10;

/** Creates a listener on interaction entries, and maps interactionIds to the origin path of the interaction */
function registerInpInteractionListener(
  interactionIdToRouteNameMapping,
  latestRoute

,
) {
  const handleEntries = ({ entries }) => {
    const client = getClient();
    // We need to get the replay, user, and activeTransaction from the current scope
    // so that we can associate replay id, profile id, and a user display to the span
    const replay =
      client !== undefined && client.getIntegrationByName !== undefined
        ? (client.getIntegrationByName('Replay') )
        : undefined;
    const replayId = replay !== undefined ? replay.getReplayId() : undefined;
    // eslint-disable-next-line deprecation/deprecation
    const activeTransaction = getActiveTransaction();
    const currentScope = getCurrentScope();
    const user = currentScope !== undefined ? currentScope.getUser() : undefined;
    entries.forEach(entry => {
      if (isPerformanceEventTiming(entry)) {
        const interactionId = entry.interactionId;
        if (interactionId === undefined) {
          return;
        }
        const existingInteraction = interactionIdToRouteNameMapping[interactionId];
        const duration = entry.duration;
        const startTime = entry.startTime;
        const keys = Object.keys(interactionIdToRouteNameMapping);
        const minInteractionId =
          keys.length > 0
            ? keys.reduce((a, b) => {
                return interactionIdToRouteNameMapping[a].duration < interactionIdToRouteNameMapping[b].duration
                  ? a
                  : b;
              })
            : undefined;
        // For a first input event to be considered, we must check that an interaction event does not already exist with the same duration and start time.
        // This is also checked in the web-vitals library.
        if (entry.entryType === 'first-input') {
          const matchingEntry = keys
            .map(key => interactionIdToRouteNameMapping[key])
            .some(interaction => {
              return interaction.duration === duration && interaction.startTime === startTime;
            });
          if (matchingEntry) {
            return;
          }
        }
        // Interactions with an id of 0 and are not first-input are not valid.
        if (!interactionId) {
          return;
        }
        // If the interaction already exists, we want to use the duration of the longest entry, since that is what the INP metric uses.
        if (existingInteraction) {
          existingInteraction.duration = Math.max(existingInteraction.duration, duration);
        } else if (
          keys.length < MAX_INTERACTIONS ||
          minInteractionId === undefined ||
          duration > interactionIdToRouteNameMapping[minInteractionId].duration
        ) {
          // If the interaction does not exist, we want to add it to the mapping if there is space, or if the duration is longer than the shortest entry.
          const routeName = latestRoute.name;
          const parentContext = latestRoute.context;
          if (routeName && parentContext) {
            if (minInteractionId && Object.keys(interactionIdToRouteNameMapping).length >= MAX_INTERACTIONS) {
              // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
              delete interactionIdToRouteNameMapping[minInteractionId];
            }
            interactionIdToRouteNameMapping[interactionId] = {
              routeName,
              duration,
              parentContext,
              user,
              activeTransaction,
              replayId,
              startTime,
            };
          }
        }
      }
    });
  };
  addPerformanceInstrumentationHandler('event', handleEntries);
  addPerformanceInstrumentationHandler('first-input', handleEntries);
}

function getSource(context) {
  const sourceFromAttributes = context.attributes && context.attributes[SEMANTIC_ATTRIBUTE_SENTRY_SOURCE];
  // eslint-disable-next-line deprecation/deprecation
  const sourceFromData = context.data && context.data[SEMANTIC_ATTRIBUTE_SENTRY_SOURCE];
  // eslint-disable-next-line deprecation/deprecation
  const sourceFromMetadata = context.metadata && context.metadata.source;

  return sourceFromAttributes || sourceFromData || sourceFromMetadata;
}

export { BROWSER_TRACING_INTEGRATION_ID, browserTracingIntegration, getMetaContent, startBrowserTracingNavigationSpan, startBrowserTracingPageLoadSpan };
//# sourceMappingURL=browserTracingIntegration.js.map
