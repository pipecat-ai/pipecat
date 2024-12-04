import { dropUndefinedKeys } from '@sentry/utils';
import { DEFAULT_ENVIRONMENT } from '../constants.js';
import { getClient, getCurrentScope } from '../exports.js';
import { getRootSpan } from '../utils/getRootSpan.js';
import { spanToJSON, spanIsSampled } from '../utils/spanUtils.js';

/**
 * Creates a dynamic sampling context from a client.
 *
 * Dispatches the `createDsc` lifecycle hook as a side effect.
 */
function getDynamicSamplingContextFromClient(
  trace_id,
  client,
  scope,
) {
  const options = client.getOptions();

  const { publicKey: public_key } = client.getDsn() || {};
  // TODO(v8): Remove segment from User
  // eslint-disable-next-line deprecation/deprecation
  const { segment: user_segment } = (scope && scope.getUser()) || {};

  const dsc = dropUndefinedKeys({
    environment: options.environment || DEFAULT_ENVIRONMENT,
    release: options.release,
    user_segment,
    public_key,
    trace_id,
  }) ;

  client.emit && client.emit('createDsc', dsc);

  return dsc;
}

/**
 * A Span with a frozen dynamic sampling context.
 */

/**
 * Creates a dynamic sampling context from a span (and client and scope)
 *
 * @param span the span from which a few values like the root span name and sample rate are extracted.
 *
 * @returns a dynamic sampling context
 */
function getDynamicSamplingContextFromSpan(span) {
  const client = getClient();
  if (!client) {
    return {};
  }

  // passing emit=false here to only emit later once the DSC is actually populated
  const dsc = getDynamicSamplingContextFromClient(spanToJSON(span).trace_id || '', client, getCurrentScope());

  // TODO (v8): Remove v7FrozenDsc as a Transaction will no longer have _frozenDynamicSamplingContext
  const txn = getRootSpan(span) ;
  if (!txn) {
    return dsc;
  }

  // TODO (v8): Remove v7FrozenDsc as a Transaction will no longer have _frozenDynamicSamplingContext
  // For now we need to avoid breaking users who directly created a txn with a DSC, where this field is still set.
  // @see Transaction class constructor
  const v7FrozenDsc = txn && txn._frozenDynamicSamplingContext;
  if (v7FrozenDsc) {
    return v7FrozenDsc;
  }

  // TODO (v8): Replace txn.metadata with txn.attributes[]
  // We can't do this yet because attributes aren't always set yet.
  // eslint-disable-next-line deprecation/deprecation
  const { sampleRate: maybeSampleRate, source } = txn.metadata;
  if (maybeSampleRate != null) {
    dsc.sample_rate = `${maybeSampleRate}`;
  }

  // We don't want to have a transaction name in the DSC if the source is "url" because URLs might contain PII
  const jsonSpan = spanToJSON(txn);

  // after JSON conversion, txn.name becomes jsonSpan.description
  if (source && source !== 'url') {
    dsc.transaction = jsonSpan.description;
  }

  dsc.sampled = String(spanIsSampled(txn));

  client.emit && client.emit('createDsc', dsc);

  return dsc;
}

export { getDynamicSamplingContextFromClient, getDynamicSamplingContextFromSpan };
//# sourceMappingURL=dynamicSamplingContext.js.map
