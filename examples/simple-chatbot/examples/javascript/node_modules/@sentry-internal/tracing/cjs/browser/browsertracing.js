Object.defineProperty(exports, '__esModule', { value: true });

const core = require('@sentry/core');
const utils = require('@sentry/utils');
const debugBuild = require('../common/debug-build.js');
const backgroundtab = require('./backgroundtab.js');
const instrument = require('./instrument.js');
const index = require('./metrics/index.js');
const request = require('./request.js');
const router = require('./router.js');
const types = require('./types.js');

const BROWSER_TRACING_INTEGRATION_ID = 'BrowserTracing';

/** Options for Browser Tracing integration */

const DEFAULT_BROWSER_TRACING_OPTIONS = {
  ...core.TRACING_DEFAULTS,
  markBackgroundTransactions: true,
  routingInstrumentation: router.instrumentRoutingWithDefaults,
  startTransactionOnLocationChange: true,
  startTransactionOnPageLoad: true,
  enableLongTask: true,
  enableInp: false,
  interactionsSampleRate: 1,
  _experiments: {},
  ...request.defaultRequestInstrumentationOptions,
};

/** We store up to 10 interaction candidates max to cap memory usage. This is the same cap as getINP from web-vitals */
const MAX_INTERACTIONS = 10;

/**
 * The Browser Tracing integration automatically instruments browser pageload/navigation
 * actions as transactions, and captures requests, metrics and errors as spans.
 *
 * The integration can be configured with a variety of options, and can be extended to use
 * any routing library. This integration uses {@see IdleTransaction} to create transactions.
 *
 * @deprecated Use `browserTracingIntegration()` instead.
 */
class BrowserTracing  {
  // This class currently doesn't have a static `id` field like the other integration classes, because it prevented
  // @sentry/tracing from being treeshaken. Tree shakers do not like static fields, because they behave like side effects.
  // TODO: Come up with a better plan, than using static fields on integration classes, and use that plan on all
  // integrations.

  /** Browser Tracing integration options */

  /**
   * @inheritDoc
   */

  // eslint-disable-next-line deprecation/deprecation

   constructor(_options) {
    this.name = BROWSER_TRACING_INTEGRATION_ID;
    this._hasSetTracePropagationTargets = false;

    core.addTracingExtensions();

    if (debugBuild.DEBUG_BUILD) {
      this._hasSetTracePropagationTargets = !!(
        _options &&
        // eslint-disable-next-line deprecation/deprecation
        (_options.tracePropagationTargets || _options.tracingOrigins)
      );
    }

    this.options = {
      ...DEFAULT_BROWSER_TRACING_OPTIONS,
      ..._options,
    };

    // Special case: enableLongTask can be set in _experiments
    // TODO (v8): Remove this in v8
    if (this.options._experiments.enableLongTask !== undefined) {
      this.options.enableLongTask = this.options._experiments.enableLongTask;
    }

    // TODO (v8): remove this block after tracingOrigins is removed
    // Set tracePropagationTargets to tracingOrigins if specified by the user
    // In case both are specified, tracePropagationTargets takes precedence
    // eslint-disable-next-line deprecation/deprecation
    if (_options && !_options.tracePropagationTargets && _options.tracingOrigins) {
      // eslint-disable-next-line deprecation/deprecation
      this.options.tracePropagationTargets = _options.tracingOrigins;
    }

    this._collectWebVitals = index.startTrackingWebVitals();
    /** Stores a mapping of interactionIds from PerformanceEventTimings to the origin interaction path */
    this._interactionIdToRouteNameMapping = {};

    if (this.options.enableInp) {
      index.startTrackingINP(this._interactionIdToRouteNameMapping, this.options.interactionsSampleRate);
    }
    if (this.options.enableLongTask) {
      index.startTrackingLongTasks();
    }
    if (this.options._experiments.enableInteractions) {
      index.startTrackingInteractions();
    }

    this._latestRoute = {
      name: undefined,
      context: undefined,
    };
  }

  /**
   * @inheritDoc
   */
  // eslint-disable-next-line deprecation/deprecation
   setupOnce(_, getCurrentHub) {
    this._getCurrentHub = getCurrentHub;
    const hub = getCurrentHub();
    // eslint-disable-next-line deprecation/deprecation
    const client = hub.getClient();
    const clientOptions = client && client.getOptions();

    const {
      routingInstrumentation: instrumentRouting,
      startTransactionOnLocationChange,
      startTransactionOnPageLoad,
      markBackgroundTransactions,
      traceFetch,
      traceXHR,
      shouldCreateSpanForRequest,
      enableHTTPTimings,
      _experiments,
    } = this.options;

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
    const tracePropagationTargets = clientOptionsTracePropagationTargets || this.options.tracePropagationTargets;
    if (debugBuild.DEBUG_BUILD && this._hasSetTracePropagationTargets && clientOptionsTracePropagationTargets) {
      utils.logger.warn(
        '[Tracing] The `tracePropagationTargets` option was set in the BrowserTracing integration and top level `Sentry.init`. The top level `Sentry.init` value is being used.',
      );
    }

    instrumentRouting(
      (context) => {
        const transaction = this._createRouteTransaction(context);

        this.options._experiments.onStartRouteTransaction &&
          this.options._experiments.onStartRouteTransaction(transaction, context, getCurrentHub);

        return transaction;
      },
      startTransactionOnPageLoad,
      startTransactionOnLocationChange,
    );

    if (markBackgroundTransactions) {
      backgroundtab.registerBackgroundTabDetection();
    }

    if (_experiments.enableInteractions) {
      this._registerInteractionListener();
    }

    if (this.options.enableInp) {
      this._registerInpInteractionListener();
    }

    request.instrumentOutgoingRequests({
      traceFetch,
      traceXHR,
      tracePropagationTargets,
      shouldCreateSpanForRequest,
      enableHTTPTimings,
    });
  }

  /** Create routing idle transaction. */
   _createRouteTransaction(context) {
    if (!this._getCurrentHub) {
      debugBuild.DEBUG_BUILD &&
        utils.logger.warn(`[Tracing] Did not create ${context.op} transaction because _getCurrentHub is invalid.`);
      return undefined;
    }

    const hub = this._getCurrentHub();

    const { beforeNavigate, idleTimeout, finalTimeout, heartbeatInterval } = this.options;

    const isPageloadTransaction = context.op === 'pageload';

    let expandedContext;
    if (isPageloadTransaction) {
      const sentryTrace = isPageloadTransaction ? getMetaContent('sentry-trace') : '';
      const baggage = isPageloadTransaction ? getMetaContent('baggage') : undefined;
      const { traceId, dsc, parentSpanId, sampled } = utils.propagationContextFromHeaders(sentryTrace, baggage);
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

    const modifiedContext = typeof beforeNavigate === 'function' ? beforeNavigate(expandedContext) : expandedContext;

    // For backwards compatibility reasons, beforeNavigate can return undefined to "drop" the transaction (prevent it
    // from being sent to Sentry).
    const finalContext = modifiedContext === undefined ? { ...expandedContext, sampled: false } : modifiedContext;

    // If `beforeNavigate` set a custom name, record that fact
    // eslint-disable-next-line deprecation/deprecation
    finalContext.metadata =
      finalContext.name !== expandedContext.name
        ? // eslint-disable-next-line deprecation/deprecation
          { ...finalContext.metadata, source: 'custom' }
        : // eslint-disable-next-line deprecation/deprecation
          finalContext.metadata;

    this._latestRoute.name = finalContext.name;
    this._latestRoute.context = finalContext;

    // eslint-disable-next-line deprecation/deprecation
    if (finalContext.sampled === false) {
      debugBuild.DEBUG_BUILD && utils.logger.log(`[Tracing] Will not send ${finalContext.op} transaction because of beforeNavigate.`);
    }

    debugBuild.DEBUG_BUILD && utils.logger.log(`[Tracing] Starting ${finalContext.op} transaction on scope`);

    const { location } = types.WINDOW;

    const idleTransaction = core.startIdleTransaction(
      hub,
      finalContext,
      idleTimeout,
      finalTimeout,
      true,
      { location }, // for use in the tracesSampler
      heartbeatInterval,
      isPageloadTransaction, // should wait for finish signal if it's a pageload transaction
    );

    if (isPageloadTransaction) {
      if (types.WINDOW.document) {
        types.WINDOW.document.addEventListener('readystatechange', () => {
          if (['interactive', 'complete'].includes(types.WINDOW.document.readyState)) {
            idleTransaction.sendAutoFinishSignal();
          }
        });

        if (['interactive', 'complete'].includes(types.WINDOW.document.readyState)) {
          idleTransaction.sendAutoFinishSignal();
        }
      }
    }

    idleTransaction.registerBeforeFinishCallback(transaction => {
      this._collectWebVitals();
      index.addPerformanceEntries(transaction);
    });

    return idleTransaction ;
  }

  /** Start listener for interaction transactions */
   _registerInteractionListener() {
    let inflightInteractionTransaction;
    const registerInteractionTransaction = () => {
      const { idleTimeout, finalTimeout, heartbeatInterval } = this.options;
      const op = 'ui.action.click';

      // eslint-disable-next-line deprecation/deprecation
      const currentTransaction = core.getActiveTransaction();
      if (currentTransaction && currentTransaction.op && ['navigation', 'pageload'].includes(currentTransaction.op)) {
        debugBuild.DEBUG_BUILD &&
          utils.logger.warn(
            `[Tracing] Did not create ${op} transaction because a pageload or navigation transaction is in progress.`,
          );
        return undefined;
      }

      if (inflightInteractionTransaction) {
        inflightInteractionTransaction.setFinishReason('interactionInterrupted');
        inflightInteractionTransaction.end();
        inflightInteractionTransaction = undefined;
      }

      if (!this._getCurrentHub) {
        debugBuild.DEBUG_BUILD && utils.logger.warn(`[Tracing] Did not create ${op} transaction because _getCurrentHub is invalid.`);
        return undefined;
      }

      if (!this._latestRoute.name) {
        debugBuild.DEBUG_BUILD && utils.logger.warn(`[Tracing] Did not create ${op} transaction because _latestRouteName is missing.`);
        return undefined;
      }

      const hub = this._getCurrentHub();
      const { location } = types.WINDOW;

      const context = {
        name: this._latestRoute.name,
        op,
        trimEnd: true,
        data: {
          [core.SEMANTIC_ATTRIBUTE_SENTRY_SOURCE]: this._latestRoute.context
            ? getSource(this._latestRoute.context)
            : 'url',
        },
      };

      inflightInteractionTransaction = core.startIdleTransaction(
        hub,
        context,
        idleTimeout,
        finalTimeout,
        true,
        { location }, // for use in the tracesSampler
        heartbeatInterval,
      );
    };

    ['click'].forEach(type => {
      if (types.WINDOW.document) {
        addEventListener(type, registerInteractionTransaction, { once: false, capture: true });
      }
    });
  }

  /** Creates a listener on interaction entries, and maps interactionIds to the origin path of the interaction */
   _registerInpInteractionListener() {
    const handleEntries = ({ entries }) => {
      const client = core.getClient();
      // We need to get the replay, user, and activeTransaction from the current scope
      // so that we can associate replay id, profile id, and a user display to the span
      const replay =
        client !== undefined && client.getIntegrationByName !== undefined
          ? (client.getIntegrationByName('Replay') )
          : undefined;
      const replayId = replay !== undefined ? replay.getReplayId() : undefined;
      // eslint-disable-next-line deprecation/deprecation
      const activeTransaction = core.getActiveTransaction();
      const currentScope = core.getCurrentScope();
      const user = currentScope !== undefined ? currentScope.getUser() : undefined;
      entries.forEach(entry => {
        if (isPerformanceEventTiming(entry)) {
          const interactionId = entry.interactionId;
          if (interactionId === undefined) {
            return;
          }
          const existingInteraction = this._interactionIdToRouteNameMapping[interactionId];
          const duration = entry.duration;
          const startTime = entry.startTime;
          const keys = Object.keys(this._interactionIdToRouteNameMapping);
          const minInteractionId =
            keys.length > 0
              ? keys.reduce((a, b) => {
                  return this._interactionIdToRouteNameMapping[a].duration <
                    this._interactionIdToRouteNameMapping[b].duration
                    ? a
                    : b;
                })
              : undefined;
          // For a first input event to be considered, we must check that an interaction event does not already exist with the same duration and start time.
          // This is also checked in the web-vitals library.
          if (entry.entryType === 'first-input') {
            const matchingEntry = keys
              .map(key => this._interactionIdToRouteNameMapping[key])
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
            duration > this._interactionIdToRouteNameMapping[minInteractionId].duration
          ) {
            // If the interaction does not exist, we want to add it to the mapping if there is space, or if the duration is longer than the shortest entry.
            const routeName = this._latestRoute.name;
            const parentContext = this._latestRoute.context;
            if (routeName && parentContext) {
              if (minInteractionId && Object.keys(this._interactionIdToRouteNameMapping).length >= MAX_INTERACTIONS) {
                // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
                delete this._interactionIdToRouteNameMapping[minInteractionId];
              }
              this._interactionIdToRouteNameMapping[interactionId] = {
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
    instrument.addPerformanceInstrumentationHandler('event', handleEntries);
    instrument.addPerformanceInstrumentationHandler('first-input', handleEntries);
  }
}

/** Returns the value of a meta tag */
function getMetaContent(metaName) {
  // Can't specify generic to `getDomElement` because tracing can be used
  // in a variety of environments, have to disable `no-unsafe-member-access`
  // as a result.
  const metaTag = utils.getDomElement(`meta[name=${metaName}]`);
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  return metaTag ? metaTag.getAttribute('content') : undefined;
}

function getSource(context) {
  const sourceFromAttributes = context.attributes && context.attributes[core.SEMANTIC_ATTRIBUTE_SENTRY_SOURCE];
  // eslint-disable-next-line deprecation/deprecation
  const sourceFromData = context.data && context.data[core.SEMANTIC_ATTRIBUTE_SENTRY_SOURCE];
  // eslint-disable-next-line deprecation/deprecation
  const sourceFromMetadata = context.metadata && context.metadata.source;

  return sourceFromAttributes || sourceFromData || sourceFromMetadata;
}

function isPerformanceEventTiming(entry) {
  return 'duration' in entry;
}

exports.BROWSER_TRACING_INTEGRATION_ID = BROWSER_TRACING_INTEGRATION_ID;
exports.BrowserTracing = BrowserTracing;
exports.getMetaContent = getMetaContent;
//# sourceMappingURL=browsertracing.js.map
