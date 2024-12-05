import { _nullishCoalesce } from './_nullishCoalesce.js';

// https://github.com/alangpierce/sucrase/tree/265887868966917f3b924ce38dfad01fbab1329f

/**
 * Polyfill for the nullish coalescing operator (`??`), when used in situations where at least one of the values is the
 * result of an async operation.
 *
 * Note that the RHS is wrapped in a function so that if it's a computed value, that evaluation won't happen unless the
 * LHS evaluates to a nullish value, to mimic the operator's short-circuiting behavior.
 *
 * Adapted from Sucrase (https://github.com/alangpierce/sucrase)
 *
 * @param lhs The value of the expression to the left of the `??`
 * @param rhsFn A function returning the value of the expression to the right of the `??`
 * @returns The LHS value, unless it's `null` or `undefined`, in which case, the RHS value
 */
async function _asyncNullishCoalesce(lhs, rhsFn) {
  return _nullishCoalesce(lhs, rhsFn);
}

// Sucrase version:
// async function _asyncNullishCoalesce(lhs, rhsFn) {
//   if (lhs != null) {
//     return lhs;
//   } else {
//     return await rhsFn();
//   }
// }

export { _asyncNullishCoalesce };
//# sourceMappingURL=_asyncNullishCoalesce.js.map
