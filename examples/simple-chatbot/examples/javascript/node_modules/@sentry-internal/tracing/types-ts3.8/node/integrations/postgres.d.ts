import { Hub } from '@sentry/core';
import { EventProcessor } from '@sentry/types';
import { LazyLoadedIntegration } from './lazy';
type PgClientQuery = (config: unknown, values?: unknown, callback?: (err: unknown, result: unknown) => void) => void | Promise<unknown>;
interface PgClient {
    prototype: {
        query: PgClientQuery;
    };
}
interface PgOptions {
    usePgNative?: boolean;
    /**
     * Supply your postgres module directly, instead of having Sentry attempt automatic resolution.
     * Use this if you (a) use a module that's not `pg`, or (b) use a bundler that breaks resolution (e.g. esbuild).
     *
     * Usage:
     * ```
     * import pg from 'pg';
     *
     * Sentry.init({
     *   integrations: [new Sentry.Integrations.Postgres({ module: pg })],
     * });
     * ```
     */
    module?: PGModule;
}
type PGModule = {
    Client: PgClient;
    native: {
        Client: PgClient;
    } | null;
};
/** Tracing integration for node-postgres package */
export declare class Postgres implements LazyLoadedIntegration<PGModule> {
    /**
     * @inheritDoc
     */
    static id: string;
    /**
     * @inheritDoc
     */
    name: string;
    private _usePgNative;
    private _module?;
    constructor(options?: PgOptions);
    /** @inheritdoc */
    loadDependency(): PGModule | undefined;
    /**
     * @inheritDoc
     */
    setupOnce(_: (callback: EventProcessor) => void, getCurrentHub: () => Hub): void;
}
export {};
//# sourceMappingURL=postgres.d.ts.map
