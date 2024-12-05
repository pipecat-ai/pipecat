import { Span } from '@sentry/types';
/** The status of an Span.
 *
 * @deprecated Use string literals - if you require type casting, cast to SpanStatusType type
 */
export declare enum SpanStatus {
    /** The operation completed successfully. */
    Ok = "ok",
    /** Deadline expired before operation could complete. */
    DeadlineExceeded = "deadline_exceeded",
    /** 401 Unauthorized (actually does mean unauthenticated according to RFC 7235) */
    Unauthenticated = "unauthenticated",
    /** 403 Forbidden */
    PermissionDenied = "permission_denied",
    /** 404 Not Found. Some requested entity (file or directory) was not found. */
    NotFound = "not_found",
    /** 429 Too Many Requests */
    ResourceExhausted = "resource_exhausted",
    /** Client specified an invalid argument. 4xx. */
    InvalidArgument = "invalid_argument",
    /** 501 Not Implemented */
    Unimplemented = "unimplemented",
    /** 503 Service Unavailable */
    Unavailable = "unavailable",
    /** Other/generic 5xx. */
    InternalError = "internal_error",
    /** Unknown. Any non-standard HTTP status code. */
    UnknownError = "unknown_error",
    /** The operation was cancelled (typically by the user). */
    Cancelled = "cancelled",
    /** Already exists (409) */
    AlreadyExists = "already_exists",
    /** Operation was rejected because the system is not in a state required for the operation's */
    FailedPrecondition = "failed_precondition",
    /** The operation was aborted, typically due to a concurrency issue. */
    Aborted = "aborted",
    /** Operation was attempted past the valid range. */
    OutOfRange = "out_of_range",
    /** Unrecoverable data loss or corruption */
    DataLoss = "data_loss"
}
export type SpanStatusType = 
/** The operation completed successfully. */
'ok'
/** Deadline expired before operation could complete. */
 | 'deadline_exceeded'
/** 401 Unauthorized (actually does mean unauthenticated according to RFC 7235) */
 | 'unauthenticated'
/** 403 Forbidden */
 | 'permission_denied'
/** 404 Not Found. Some requested entity (file or directory) was not found. */
 | 'not_found'
/** 429 Too Many Requests */
 | 'resource_exhausted'
/** Client specified an invalid argument. 4xx. */
 | 'invalid_argument'
/** 501 Not Implemented */
 | 'unimplemented'
/** 503 Service Unavailable */
 | 'unavailable'
/** Other/generic 5xx. */
 | 'internal_error'
/** Unknown. Any non-standard HTTP status code. */
 | 'unknown_error'
/** The operation was cancelled (typically by the user). */
 | 'cancelled'
/** Already exists (409) */
 | 'already_exists'
/** Operation was rejected because the system is not in a state required for the operation's */
 | 'failed_precondition'
/** The operation was aborted, typically due to a concurrency issue. */
 | 'aborted'
/** Operation was attempted past the valid range. */
 | 'out_of_range'
/** Unrecoverable data loss or corruption */
 | 'data_loss';
/**
 * Converts a HTTP status code into a {@link SpanStatusType}.
 *
 * @param httpStatus The HTTP response status code.
 * @returns The span status or unknown_error.
 */
export declare function getSpanStatusFromHttpCode(httpStatus: number): SpanStatusType;
/**
 * Converts a HTTP status code into a {@link SpanStatusType}.
 *
 * @deprecated Use {@link spanStatusFromHttpCode} instead.
 * This export will be removed in v8 as the signature contains a typo.
 *
 * @param httpStatus The HTTP response status code.
 * @returns The span status or unknown_error.
 */
export declare const spanStatusfromHttpCode: typeof getSpanStatusFromHttpCode;
/**
 * Sets the Http status attributes on the current span based on the http code.
 * Additionally, the span's status is updated, depending on the http code.
 */
export declare function setHttpStatus(span: Span, httpStatus: number): void;
//# sourceMappingURL=spanstatus.d.ts.map
