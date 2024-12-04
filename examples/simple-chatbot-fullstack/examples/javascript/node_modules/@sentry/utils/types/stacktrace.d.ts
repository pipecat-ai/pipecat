import type { StackFrame, StackLineParser, StackParser } from '@sentry/types';
import type { GetModuleFn } from './node-stack-trace';
import { filenameIsInApp } from './node-stack-trace';
export { filenameIsInApp };
/**
 * Creates a stack parser with the supplied line parsers
 *
 * StackFrames are returned in the correct order for Sentry Exception
 * frames and with Sentry SDK internal frames removed from the top and bottom
 *
 */
export declare function createStackParser(...parsers: StackLineParser[]): StackParser;
/**
 * Gets a stack parser implementation from Options.stackParser
 * @see Options
 *
 * If options contains an array of line parsers, it is converted into a parser
 */
export declare function stackParserFromStackParserOptions(stackParser: StackParser | StackLineParser[]): StackParser;
/**
 * Removes Sentry frames from the top and bottom of the stack if present and enforces a limit of max number of frames.
 * Assumes stack input is ordered from top to bottom and returns the reverse representation so call site of the
 * function that caused the crash is the last frame in the array.
 * @hidden
 */
export declare function stripSentryFramesAndReverse(stack: ReadonlyArray<StackFrame>): StackFrame[];
/**
 * Safely extract function name from itself
 */
export declare function getFunctionName(fn: unknown): string;
/**
 * Node.js stack line parser
 *
 * This is in @sentry/utils so it can be used from the Electron SDK in the browser for when `nodeIntegration == true`.
 * This allows it to be used without referencing or importing any node specific code which causes bundlers to complain
 */
export declare function nodeStackLineParser(getModule?: GetModuleFn): StackLineParser;
//# sourceMappingURL=stacktrace.d.ts.map