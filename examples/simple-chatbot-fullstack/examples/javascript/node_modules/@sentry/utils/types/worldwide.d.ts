/**
 * NOTE: In order to avoid circular dependencies, if you add a function to this module and it needs to print something,
 * you must either a) use `console.log` rather than the logger, or b) put your function elsewhere.
 *
 * Note: This file was originally called `global.ts`, but was changed to unblock users which might be doing
 * string replaces with bundlers like Vite for `global` (would break imports that rely on importing from utils/src/global).
 *
 * Why worldwide?
 *
 * Why not?
 */
import type { Integration } from '@sentry/types';
import type { SdkSource } from './env';
/** Internal global with common properties and Sentry extensions  */
export interface InternalGlobal {
    navigator?: {
        userAgent?: string;
    };
    console: Console;
    Sentry?: {
        Integrations?: Integration[];
    };
    onerror?: {
        (event: object | string, source?: string, lineno?: number, colno?: number, error?: Error): any;
        __SENTRY_INSTRUMENTED__?: true;
        __SENTRY_LOADER__?: true;
    };
    onunhandledrejection?: {
        (event: unknown): boolean;
        __SENTRY_INSTRUMENTED__?: true;
        __SENTRY_LOADER__?: true;
    };
    SENTRY_ENVIRONMENT?: string;
    SENTRY_DSN?: string;
    SENTRY_RELEASE?: {
        id?: string;
    };
    SENTRY_SDK_SOURCE?: SdkSource;
    /**
     * Debug IDs are indirectly injected by Sentry CLI or bundler plugins to directly reference a particular source map
     * for resolving of a source file. The injected code will place an entry into the record for each loaded bundle/JS
     * file.
     */
    _sentryDebugIds?: Record<string, string>;
    __SENTRY__: {
        globalEventProcessors: any;
        hub: any;
        logger: any;
        extensions?: {
            /** Extension methods for the hub, which are bound to the current Hub instance */
            [key: string]: Function;
        };
    };
    /**
     * Raw module metadata that is injected by bundler plugins.
     *
     * Keys are `error.stack` strings, values are the metadata.
     */
    _sentryModuleMetadata?: Record<string, any>;
}
/** Get's the global object for the current JavaScript runtime */
export declare const GLOBAL_OBJ: InternalGlobal;
/**
 * @deprecated Use GLOBAL_OBJ instead or WINDOW from @sentry/browser. This will be removed in v8
 */
export declare function getGlobalObject<T>(): T & InternalGlobal;
/**
 * Returns a global singleton contained in the global `__SENTRY__` object.
 *
 * If the singleton doesn't already exist in `__SENTRY__`, it will be created using the given factory
 * function and added to the `__SENTRY__` object.
 *
 * @param name name of the global singleton on __SENTRY__
 * @param creator creator Factory function to create the singleton if it doesn't already exist on `__SENTRY__`
 * @param obj (Optional) The global object on which to look for `__SENTRY__`, if not `GLOBAL_OBJ`'s return value
 * @returns the singleton
 */
export declare function getGlobalSingleton<T>(name: keyof InternalGlobal['__SENTRY__'], creator: () => T, obj?: unknown): T;
//# sourceMappingURL=worldwide.d.ts.map