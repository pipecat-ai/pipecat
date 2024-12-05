Object.defineProperty(exports, '__esModule', { value: true });

const exports$1 = require('../exports.js');

// Treeshakable guard to remove all code related to tracing

/**
 * Determines if tracing is currently enabled.
 *
 * Tracing is enabled when at least one of `tracesSampleRate` and `tracesSampler` is defined in the SDK config.
 */
function hasTracingEnabled(
  maybeOptions,
) {
  if (typeof __SENTRY_TRACING__ === 'boolean' && !__SENTRY_TRACING__) {
    return false;
  }

  const client = exports$1.getClient();
  const options = maybeOptions || (client && client.getOptions());
  return !!options && (options.enableTracing || 'tracesSampleRate' in options || 'tracesSampler' in options);
}

exports.hasTracingEnabled = hasTracingEnabled;
//# sourceMappingURL=hasTracingEnabled.js.map
