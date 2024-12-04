import type { ConsoleLevel } from '@sentry/types';
export declare const CONSOLE_LEVELS: readonly ConsoleLevel[];
type LoggerMethod = (...args: unknown[]) => void;
type LoggerConsoleMethods = Record<ConsoleLevel, LoggerMethod>;
/** This may be mutated by the console instrumentation. */
export declare const originalConsoleMethods: {
    [key in ConsoleLevel]?: (...args: any[]) => void;
};
/** JSDoc */
interface Logger extends LoggerConsoleMethods {
    disable(): void;
    enable(): void;
    isEnabled(): boolean;
}
/**
 * Temporarily disable sentry console instrumentations.
 *
 * @param callback The function to run against the original `console` messages
 * @returns The results of the callback
 */
export declare function consoleSandbox<T>(callback: () => T): T;
export declare const logger: Logger;
export {};
//# sourceMappingURL=logger.d.ts.map