import { _optionalChain } from '@sentry/utils';

/**
 * Check if Sentry auto-instrumentation should be disabled.
 *
 * @param getCurrentHub A method to fetch the current hub
 * @returns boolean
 */
// eslint-disable-next-line deprecation/deprecation
function shouldDisableAutoInstrumentation(getCurrentHub) {
  // eslint-disable-next-line deprecation/deprecation
  const clientOptions = _optionalChain([getCurrentHub, 'call', _ => _(), 'access', _2 => _2.getClient, 'call', _3 => _3(), 'optionalAccess', _4 => _4.getOptions, 'call', _5 => _5()]);
  const instrumenter = _optionalChain([clientOptions, 'optionalAccess', _6 => _6.instrumenter]) || 'sentry';

  return instrumenter !== 'sentry';
}

export { shouldDisableAutoInstrumentation };
//# sourceMappingURL=node-utils.js.map
