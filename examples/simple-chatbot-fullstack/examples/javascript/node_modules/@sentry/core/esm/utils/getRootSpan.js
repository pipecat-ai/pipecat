/**
 * Returns the root span of a given span.
 *
 * As long as we use `Transaction`s internally, the returned root span
 * will be a `Transaction` but be aware that this might change in the future.
 *
 * If the given span has no root span or transaction, `undefined` is returned.
 */
function getRootSpan(span) {
  // TODO (v8): Remove this check and just return span
  // eslint-disable-next-line deprecation/deprecation
  return span.transaction;
}

export { getRootSpan };
//# sourceMappingURL=getRootSpan.js.map
