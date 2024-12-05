import { tracingContextFromHeaders, logger, dropUndefinedKeys, addNonEnumerableProperty } from '@sentry/utils';
import { DEBUG_BUILD } from '../debug-build.js';
import { getCurrentHub, runWithAsyncContext, getIsolationScope } from '../hub.js';
import { spanToJSON, spanIsSampled, spanTimeInputToSeconds } from '../utils/spanUtils.js';
import './errors.js';
import './spanstatus.js';
import { getDynamicSamplingContextFromSpan } from './dynamicSamplingContext.js';
import { getCurrentScope, withScope } from '../exports.js';
import { handleCallbackErrors } from '../utils/handleCallbackErrors.js';
import { hasTracingEnabled } from '../utils/hasTracingEnabled.js';

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
  const hub = getCurrentHub();
  const scope = getCurrentScope();
  // eslint-disable-next-line deprecation/deprecation
  const parentSpan = scope.getSpan();

  const spanContext = normalizeContext(context);
  const activeSpan = createChildSpanOrTransaction(hub, {
    parentSpan,
    spanContext,
    forceTransaction: false,
    scope,
  });

  // eslint-disable-next-line deprecation/deprecation
  scope.setSpan(activeSpan);

  return handleCallbackErrors(
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

  return runWithAsyncContext(() => {
    return withScope(context.scope, scope => {
      // eslint-disable-next-line deprecation/deprecation
      const hub = getCurrentHub();
      // eslint-disable-next-line deprecation/deprecation
      const parentSpan = scope.getSpan();

      const shouldSkipSpan = context.onlyIfParent && !parentSpan;
      const activeSpan = shouldSkipSpan
        ? undefined
        : createChildSpanOrTransaction(hub, {
            parentSpan,
            spanContext,
            forceTransaction: context.forceTransaction,
            scope,
          });

      return handleCallbackErrors(
        () => callback(activeSpan),
        () => {
          // Only update the span status if it hasn't been changed yet
          if (activeSpan) {
            const { status } = spanToJSON(activeSpan);
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

  return runWithAsyncContext(() => {
    return withScope(context.scope, scope => {
      // eslint-disable-next-line deprecation/deprecation
      const hub = getCurrentHub();
      // eslint-disable-next-line deprecation/deprecation
      const parentSpan = scope.getSpan();

      const shouldSkipSpan = context.onlyIfParent && !parentSpan;
      const activeSpan = shouldSkipSpan
        ? undefined
        : createChildSpanOrTransaction(hub, {
            parentSpan,
            spanContext,
            forceTransaction: context.forceTransaction,
            scope,
          });

      function finishAndSetSpan() {
        activeSpan && activeSpan.end();
      }

      return handleCallbackErrors(
        () => callback(activeSpan, finishAndSetSpan),
        () => {
          // Only update the span status if it hasn't been changed yet, and the span is not yet finished
          if (activeSpan && activeSpan.isRecording()) {
            const { status } = spanToJSON(activeSpan);
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
  if (!hasTracingEnabled()) {
    return undefined;
  }

  const spanContext = normalizeContext(context);
  // eslint-disable-next-line deprecation/deprecation
  const hub = getCurrentHub();
  const parentSpan = context.scope
    ? // eslint-disable-next-line deprecation/deprecation
      context.scope.getSpan()
    : getActiveSpan();

  const shouldSkipSpan = context.onlyIfParent && !parentSpan;

  if (shouldSkipSpan) {
    return undefined;
  }

  const scope = context.scope || getCurrentScope();

  // Even though we don't actually want to make this span active on the current scope,
  // we need to make it active on a temporary scope that we use for event processing
  // as otherwise, it won't pick the correct span for the event when processing it
  const temporaryScope = (scope ).clone();

  return createChildSpanOrTransaction(hub, {
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
  return getCurrentScope().getSpan();
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

  const currentScope = getCurrentScope();

  // eslint-disable-next-line deprecation/deprecation
  const { traceparentData, dynamicSamplingContext, propagationContext } = tracingContextFromHeaders(
    sentryTrace,
    baggage,
  );

  currentScope.setPropagationContext(propagationContext);

  if (DEBUG_BUILD && traceparentData) {
    logger.log(`[Tracing] Continuing trace ${traceparentData.traceId}.`);
  }

  const transactionContext = {
    ...traceparentData,
    metadata: dropUndefinedKeys({
      dynamicSamplingContext,
    }),
  };

  if (!callback) {
    return transactionContext;
  }

  return runWithAsyncContext(() => {
    return callback(transactionContext);
  });
};

function createChildSpanOrTransaction(
  // eslint-disable-next-line deprecation/deprecation
  hub,
  {
    parentSpan,
    spanContext,
    forceTransaction,
    scope,
  }

,
) {
  if (!hasTracingEnabled()) {
    return undefined;
  }

  const isolationScope = getIsolationScope();

  let span;
  if (parentSpan && !forceTransaction) {
    // eslint-disable-next-line deprecation/deprecation
    span = parentSpan.startChild(spanContext);
  } else if (parentSpan) {
    // If we forced a transaction but have a parent span, make sure to continue from the parent span, not the scope
    const dsc = getDynamicSamplingContextFromSpan(parentSpan);
    const { traceId, spanId: parentSpanId } = parentSpan.spanContext();
    const sampled = spanIsSampled(parentSpan);

    // eslint-disable-next-line deprecation/deprecation
    span = hub.startTransaction({
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
    span = hub.startTransaction({
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
    ctx.startTimestamp = spanTimeInputToSeconds(context.startTime);
    delete ctx.startTime;
    return ctx;
  }

  return context;
}

const SCOPE_ON_START_SPAN_FIELD = '_sentryScope';
const ISOLATION_SCOPE_ON_START_SPAN_FIELD = '_sentryIsolationScope';

function setCapturedScopesOnSpan(span, scope, isolationScope) {
  if (span) {
    addNonEnumerableProperty(span, ISOLATION_SCOPE_ON_START_SPAN_FIELD, isolationScope);
    addNonEnumerableProperty(span, SCOPE_ON_START_SPAN_FIELD, scope);
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

export { continueTrace, getActiveSpan, getCapturedScopesOnSpan, startActiveSpan, startInactiveSpan, startSpan, startSpanManual, trace };
//# sourceMappingURL=trace.js.map
