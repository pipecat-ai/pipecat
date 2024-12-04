/** The status of an Span.
 *
 * @deprecated Use string literals - if you require type casting, cast to SpanStatusType type
 */
var SpanStatus; (function (SpanStatus) {
  /** The operation completed successfully. */
  const Ok = 'ok'; SpanStatus["Ok"] = Ok;
  /** Deadline expired before operation could complete. */
  const DeadlineExceeded = 'deadline_exceeded'; SpanStatus["DeadlineExceeded"] = DeadlineExceeded;
  /** 401 Unauthorized (actually does mean unauthenticated according to RFC 7235) */
  const Unauthenticated = 'unauthenticated'; SpanStatus["Unauthenticated"] = Unauthenticated;
  /** 403 Forbidden */
  const PermissionDenied = 'permission_denied'; SpanStatus["PermissionDenied"] = PermissionDenied;
  /** 404 Not Found. Some requested entity (file or directory) was not found. */
  const NotFound = 'not_found'; SpanStatus["NotFound"] = NotFound;
  /** 429 Too Many Requests */
  const ResourceExhausted = 'resource_exhausted'; SpanStatus["ResourceExhausted"] = ResourceExhausted;
  /** Client specified an invalid argument. 4xx. */
  const InvalidArgument = 'invalid_argument'; SpanStatus["InvalidArgument"] = InvalidArgument;
  /** 501 Not Implemented */
  const Unimplemented = 'unimplemented'; SpanStatus["Unimplemented"] = Unimplemented;
  /** 503 Service Unavailable */
  const Unavailable = 'unavailable'; SpanStatus["Unavailable"] = Unavailable;
  /** Other/generic 5xx. */
  const InternalError = 'internal_error'; SpanStatus["InternalError"] = InternalError;
  /** Unknown. Any non-standard HTTP status code. */
  const UnknownError = 'unknown_error'; SpanStatus["UnknownError"] = UnknownError;
  /** The operation was cancelled (typically by the user). */
  const Cancelled = 'cancelled'; SpanStatus["Cancelled"] = Cancelled;
  /** Already exists (409) */
  const AlreadyExists = 'already_exists'; SpanStatus["AlreadyExists"] = AlreadyExists;
  /** Operation was rejected because the system is not in a state required for the operation's */
  const FailedPrecondition = 'failed_precondition'; SpanStatus["FailedPrecondition"] = FailedPrecondition;
  /** The operation was aborted, typically due to a concurrency issue. */
  const Aborted = 'aborted'; SpanStatus["Aborted"] = Aborted;
  /** Operation was attempted past the valid range. */
  const OutOfRange = 'out_of_range'; SpanStatus["OutOfRange"] = OutOfRange;
  /** Unrecoverable data loss or corruption */
  const DataLoss = 'data_loss'; SpanStatus["DataLoss"] = DataLoss;
})(SpanStatus || (SpanStatus = {}));

/**
 * Converts a HTTP status code into a {@link SpanStatusType}.
 *
 * @param httpStatus The HTTP response status code.
 * @returns The span status or unknown_error.
 */
function getSpanStatusFromHttpCode(httpStatus) {
  if (httpStatus < 400 && httpStatus >= 100) {
    return 'ok';
  }

  if (httpStatus >= 400 && httpStatus < 500) {
    switch (httpStatus) {
      case 401:
        return 'unauthenticated';
      case 403:
        return 'permission_denied';
      case 404:
        return 'not_found';
      case 409:
        return 'already_exists';
      case 413:
        return 'failed_precondition';
      case 429:
        return 'resource_exhausted';
      default:
        return 'invalid_argument';
    }
  }

  if (httpStatus >= 500 && httpStatus < 600) {
    switch (httpStatus) {
      case 501:
        return 'unimplemented';
      case 503:
        return 'unavailable';
      case 504:
        return 'deadline_exceeded';
      default:
        return 'internal_error';
    }
  }

  return 'unknown_error';
}

/**
 * Converts a HTTP status code into a {@link SpanStatusType}.
 *
 * @deprecated Use {@link spanStatusFromHttpCode} instead.
 * This export will be removed in v8 as the signature contains a typo.
 *
 * @param httpStatus The HTTP response status code.
 * @returns The span status or unknown_error.
 */
const spanStatusfromHttpCode = getSpanStatusFromHttpCode;

/**
 * Sets the Http status attributes on the current span based on the http code.
 * Additionally, the span's status is updated, depending on the http code.
 */
function setHttpStatus(span, httpStatus) {
  // TODO (v8): Remove these calls
  // Relay does not require us to send the status code as a tag
  // For now, just because users might expect it to land as a tag we keep sending it.
  // Same with data.
  // In v8, we replace both, simply with
  // span.setAttribute('http.response.status_code', httpStatus);

  // eslint-disable-next-line deprecation/deprecation
  span.setTag('http.status_code', String(httpStatus));
  // eslint-disable-next-line deprecation/deprecation
  span.setData('http.response.status_code', httpStatus);

  const spanStatus = getSpanStatusFromHttpCode(httpStatus);
  if (spanStatus !== 'unknown_error') {
    span.setStatus(spanStatus);
  }
}

export { SpanStatus, getSpanStatusFromHttpCode, setHttpStatus, spanStatusfromHttpCode };
//# sourceMappingURL=spanstatus.js.map
