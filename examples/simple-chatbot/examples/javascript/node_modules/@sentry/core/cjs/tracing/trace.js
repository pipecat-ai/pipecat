Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const debugBuild = require('../debug-build.js');
const hub = require('../hub.js');
const spanUtils = require('../utils/spanUtils.js');
require('./errors.js');
require('./spanstatus.js');
const dynamicSamplingContext = require('./dynamicSamplingContext.js');
const exports$1 = require('../exports.js');
const handleCallbackErrors = require('../utils/handleCallbackErrors.js');
const hasTracingEnabled = require('../utils/hasTracingEnabled.js');

/**
 * Wraps a function with a transaction/span and finishes the span after the function is done.
 *
 * Note that if you have not enabled tracing extensions via `addTracingExtensions`
 * or you didn't set `tracesSampleRate`, this function will not generate spans
 * and the `span` returned from the callback will be undefined.
 *
 * This function is meant to be used internally and may break at any time. Use at your own risk.
 *
 * @internal
 * @private
 *
 * @deprecated Use `startSpan` instead.
 */
function trace(
  context,
  callback,
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  onError = () => {},
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  afterFinish = () => {},
) {
  // eslint-disable-next-line deprecation/deprecation
  const hub$1 = hub.getCurrentHub();
  const scope = exports$1.getCurrentScope();
  // eslint-disable-next-line deprecation/deprecation
  const parentSpan = scope.getSpan();

  const spanContext = normalizeContext(context);
  const activeSpan = createChildSpanOrTransaction(hub$1, {
    parentSpan,
    spanContext,
    forceTransaction: false,
    scope,
  });

  // eslint-disable-next-line deprecation/deprecation
  scope.setSpan(activeSpan);

  return handleCallbackErrors.handleCallbackErrors(
    () => callback(activeSpan),
    error => {
      activeSpan && activeSpan.setStatus('internal_error');
      onError(error, activeSpan);
    },
    () => {
      activeSpan && activeSpan.end();
      // eslint-disable-next-line deprecation/deprecation
      scope.setSpan(parentSpan);
      afterFinish();
    },
  );
}

/**
 * Wraps a function with a transaction/span and finishes the span after the function is done.
 * The created span is the active span and will be used as parent by other spans created inside the function
 * and can be accessed via `Sentry.getSpan()`, as long as the function is executed while the scope is active.
 *
 * If you want to create a span that is not set as active, use {@link startInactiveSpan}.
 *
 * Note that if you have not enabled tracing extensions via `addTracingExtensions`
 * or you didn't set `tracesSampleRate`, this function will not generate spans
 * and the `span` returned from the callback will be undefined.
 */
function startSpan(context, callback) {
  const spanContext = normalizeContext(context);

  return hub.runWithAsyncContext(() => {
    return exports$1.withScope(context.scope, scope => {
      // eslint-disable-next-line deprecation/deprecation
      const hub$1 = hub.getCurrentHub();
      // eslint-disable-next-line deprecation/deprecation
      const parentSpan = scope.getSpan();

      const shouldSkipSpan = context.onlyIfParent && !parentSpan;
      const activeSpan = shouldSkipSpan
        ? undefined
        : createChildSpanOrTransaction(hub$1, {
            parentSpan,
            spanContext,
            forceTransaction: context.forceTransaction,
            scope,
          });

      return handleCallbackErrors.handleCallbackErrors(
        () => callback(activeSpan),
        () => {
          // Only update the span status if it hasn't been changed yet
          if (activeSpan) {
            const { status } = spanUtils.spanToJSON(activeSpan);
            if (!status || status === 'ok') {
              activeSpan.setStatus('internal_error');
            }
          }
        },
        () => activeSpan && activeSpan.end(),
      );
    });
  });
}

/**
 * @deprecated Use {@link startSpan} instead.
 */
const startActiveSpan = startSpan;

/**
 * Similar to `Sentry.startSpan`. Wraps a function with a transaction/span, but does not finish the span
 * after the function is done automatically. You'll have to call `span.end()` manually.
 *
 * The created span is the active span and will be used as parent by other spans created inside the function
 * and can be accessed via `Sentry.getActiveSpan()`, as long as the function is executed while the scope is active.
 *
 * Note that if you have not enabled tracing extensions via `addTracingExtensions`
 * or you didn't set `tracesSampleRate`, this function will not generate spans
 * and the `span` returned from the callback will be undefined.
 */
function startSpanManual(
  context,
  callback,
) {
  const spanContext = normalizeContext(context);

  return hub.runWithAsyncContext(() => {
    return exports$1.withScope(context.scope, scope => {
      // eslint-disable-next-line deprecation/deprecation
      const hub$1 = hub.getCurrentHub();
      // eslint-disable-next-line deprecation/deprecation
      const parentSpan = scope.getSpan();

      const shouldSkipSpan = context.onlyIfParent && !parentSpan;
      const activeSpan = shouldSkipSpan
        ? undefined
        : createChildSpanOrTransaction(hub$1, {
            parentSpan,
            spanContext,
            forceTransaction: context.forceTransaction,
            scope,
          });

      function finishAndSetSpan() {
        activeSpan && activeSpan.end();
      }

      return handleCallbackErrors.handleCallbackErrors(
        () => callback(activeSpan, finishAndSetSpan),
        () => {
          // Only update the span status if it hasn't been changed yet, and the span is not yet finished
          if (activeSpan && activeSpan.isRecording()) {
            const { status } = spanUtils.spanToJSON(activeSpan);
            if (!status || status === 'ok') {
              activeSpan.setStatus('internal_error');
            }
          }
        },
      );
    });
  });
}

/**
 * Creates a span. This span is not set as active, so will not get automatic instrumentation spans
 * as children or be able to be accessed via `Sentry.getSpan()`.
 *
 * If you want to create a span that is set as active, use {@link startSpan}.
 *
 * Note that if you have not enabled tracing extensions via `addTracingExtensions`
 * or you didn't set `tracesSampleRate` or `tracesSampler`, this function will not generate spans
 * and the `span` returned from the callback will be undefined.
 */
function startInactiveSpan(context) {
  if (!hasTracingEnabled.hasTracingEnabled()) {
    return undefined;
  }

  const spanContext = normalizeContext(context);
  // eslint-disable-next-line deprecation/deprecation
  const hub$1 = hub.getCurrentHub();
  const parentSpan = context.scope
    ? // eslint-disable-next-line deprecation/deprecation
      context.scope.getSpan()
    : getActiveSpan();

  const shouldSkipSpan = context.onlyIfParent && !parentSpan;

  if (shouldSkipSpan) {
    return undefined;
  }

  const scope = context.scope || exports$1.getCurrentScope();

  // Even though we don't actually want to make this span active on the current scope,
  // we need to make it active on a temporary scope that we use for event processing
  // as otherwise, it won't pick the correct span for the event when processing it
  const temporaryScope = (scope ).clone();

  return createChildSpanOrTransaction(hub$1, {
    parentSpan,
    spanContext,
    forceTransaction: context.forceTransaction,
    scope: temporaryScope,
  });
}

/**
 * Returns the currently active span.
 */
function getActiveSpan() {
  // eslint-disable-next-line deprecation/deprecation
  return exports$1.getCurrentScope().getSpan();
}

const continueTrace = (
  {
    sentryTrace,
    baggage,
  }

,
  callback,
) => {
  // TODO(v8): Change this function so it doesn't do anything besides setting the propagation context on the current scope:
  /*
    return withScope((scope) => {
      const propagationContext = propagationContextFromHeaders(sentryTrace, baggage);
      scope.setPropagationContext(propagationContext);
      return callback();
    })
  */

  const currentScope = exports$1.getCurrentScope();

  // eslint-disable-next-line deprecation/deprecation
  const { traceparentData, dynamicSamplingContext, propagationContext } = utils.tracingContextFromHeaders(
    sentryTrace,
    baggage,
  );

  currentScope.setPropagationContext(propagationContext);

  if (debugBuild.DEBUG_BUILD && traceparentData) {
    utils.logger.log(`[Tracing] Continuing trace ${traceparentData.traceId}.`);
  }

  const transactionContext = {
    ...traceparentData,
    metadata: utils.dropUndefinedKeys({
      dynamicSamplingContext,
    }),
  };

  if (!callback) {
    return transactionContext;
  }

  return hub.runWithAsyncContext(() => {
    return callback(transactionContext);
  });
};

function createChildSpanOrTransaction(
  // eslint-disable-next-line deprecation/deprecation
  hub$1,
  {
    parentSpan,
    spanContext,
    forceTransaction,
    scope,
  }

,
) {
  if (!hasTracingEnabled.hasTracingEnabled()) {
    return undefined;
  }

  const isolationScope = hub.getIsolationScope();

  let span;
  if (parentSpan && !forceTransaction) {
    // eslint-disable-next-line deprecation/deprecation
    span = parentSpan.startChild(spanContext);
  } else if (parentSpan) {
    // If we forced a transaction but have a parent span, make sure to continue from the parent span, not the scope
    const dsc = dynamicSamplingContext.getDynamicSamplingContextFromSpan(parentSpan);
    const { traceId, spanId: parentSpanId } = parentSpan.spanContext();
    const sampled = spanUtils.spanIsSampled(parentSpan);

    // eslint-disable-next-line deprecation/deprecation
    span = hub$1.startTransaction({
      traceId,
      parentSpanId,
      parentSampled: sampled,
      ...spanContext,
      metadata: {
        dynamicSamplingContext: dsc,
        // eslint-disable-next-line deprecation/deprecation
        ...spanContext.metadata,
      },
    });
  } else {
    const { traceId, dsc, parentSpanId, sampled } = {
      ...isolationScope.getPropagationContext(),
      ...scope.getPropagationContext(),
    };

    // eslint-disable-next-line deprecation/deprecation
    span = hub$1.startTransaction({
      traceId,
      parentSpanId,
      parentSampled: sampled,
      ...spanContext,
      metadata: {
        dynamicSamplingContext: dsc,
        // eslint-disable-next-line deprecation/deprecation
        ...spanContext.metadata,
      },
    });
  }

  // We always set this as active span on the scope
  // In the case of this being an inactive span, we ensure to pass a detached scope in here in the first place
  // But by having this here, we can ensure that the lookup through `getCapturedScopesOnSpan` results in the correct scope & span combo
  // eslint-disable-next-line deprecation/deprecation
  scope.setSpan(span);

  setCapturedScopesOnSpan(span, scope, isolationScope);

  return span;
}

/**
 * This converts StartSpanOptions to TransactionContext.
 * For the most part (for now) we accept the same options,
 * but some of them need to be transformed.
 *
 * Eventually the StartSpanOptions will be more aligned with OpenTelemetry.
 */
function normalizeContext(context) {
  if (context.startTime) {
    const ctx = { ...context };
    ctx.startTimestamp = spanUtils.spanTimeInputToSeconds(context.startTime);
    delete ctx.startTime;
    return ctx;
  }

  return context;
}

const SCOPE_ON_START_SPAN_FIELD = '_sentryScope';
const ISOLATION_SCOPE_ON_START_SPAN_FIELD = '_sentryIsolationScope';

function setCapturedScopesOnSpan(span, scope, isolationScope) {
  if (span) {
    utils.addNonEnumerableProperty(span, ISOLATION_SCOPE_ON_START_SPAN_FIELD, isolationScope);
    utils.addNonEnumerableProperty(span, SCOPE_ON_START_SPAN_FIELD, scope);
  }
}

/**
 * Grabs the scope and isolation scope off a span that were active when the span was started.
 */
function getCapturedScopesOnSpan(span) {
  return {
    scope: (span )[SCOPE_ON_START_SPAN_FIELD],
    isolationScope: (span )[ISOLATION_SCOPE_ON_START_SPAN_FIELD],
  };
}

exports.continueTrace = continueTrace;
exports.getActiveSpan = getActiveSpan;
exports.getCapturedScopesOnSpan = getCapturedScopesOnSpan;
exports.startActiveSpan = startActiveSpan;
exports.startInactiveSpan = startInactiveSpan;
exports.startSpan = startSpan;
exports.startSpanManual = startSpanManual;
exports.trace = trace;
//# sourceMappingURL=trace.js.map
