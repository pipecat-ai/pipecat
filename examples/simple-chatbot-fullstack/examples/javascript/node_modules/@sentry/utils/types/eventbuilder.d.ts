import type { Client, Event, EventHint, Exception, Hub, ParameterizedString, Severity, SeverityLevel, StackFrame, StackParser } from '@sentry/types';
/**
 * Extracts stack frames from the error.stack string
 */
export declare function parseStackFrames(stackParser: StackParser, error: Error): StackFrame[];
/**
 * Extracts stack frames from the error and builds a Sentry Exception
 */
export declare function exceptionFromError(stackParser: StackParser, error: Error): Exception;
/**
 * Builds and Event from a Exception
 *
 * TODO(v8): Remove getHub fallback
 * @hidden
 */
export declare function eventFromUnknownInput(getHubOrClient: (() => Hub) | Client | undefined, stackParser: StackParser, exception: unknown, hint?: EventHint): Event;
/**
 * Builds and Event from a Message
 * @hidden
 */
export declare function eventFromMessage(stackParser: StackParser, message: ParameterizedString, level?: Severity | SeverityLevel, hint?: EventHint, attachStacktrace?: boolean): Event;
//# sourceMappingURL=eventbuilder.d.ts.map