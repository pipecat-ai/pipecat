import { Event, EventHint, Exception, ParameterizedString, Severity, SeverityLevel, StackFrame, StackParser } from '@sentry/types';
/**
 * This function creates an exception from a JavaScript Error
 */
export declare function exceptionFromError(stackParser: StackParser, ex: Error): Exception;
/**
 * @hidden
 */
export declare function eventFromPlainObject(stackParser: StackParser, exception: Record<string, unknown>, syntheticException?: Error, isUnhandledRejection?: boolean): Event;
/**
 * @hidden
 */
export declare function eventFromError(stackParser: StackParser, ex: Error): Event;
/** Parses stack frames from an error */
export declare function parseStackFrames(stackParser: StackParser, ex: Error & {
    framesToPop?: number;
    stacktrace?: string;
}): StackFrame[];
/**
 * Creates an {@link Event} from all inputs to `captureException` and non-primitive inputs to `captureMessage`.
 * @hidden
 */
export declare function eventFromException(stackParser: StackParser, exception: unknown, hint?: EventHint, attachStacktrace?: boolean): PromiseLike<Event>;
/**
 * Builds and Event from a Message
 * @hidden
 */
export declare function eventFromMessage(stackParser: StackParser, message: ParameterizedString, level?: Severity | SeverityLevel, hint?: EventHint, attachStacktrace?: boolean): PromiseLike<Event>;
/**
 * @hidden
 */
export declare function eventFromUnknownInput(stackParser: StackParser, exception: unknown, syntheticException?: Error, attachStacktrace?: boolean, isUnhandledRejection?: boolean): Event;
/**
 * @hidden
 */
export declare function eventFromString(stackParser: StackParser, message: ParameterizedString, syntheticException?: Error, attachStacktrace?: boolean): Event;
//# sourceMappingURL=eventbuilder.d.ts.map
