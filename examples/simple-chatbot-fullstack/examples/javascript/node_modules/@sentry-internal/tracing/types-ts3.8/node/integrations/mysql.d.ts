import { Hub } from '@sentry/core';
import { EventProcessor } from '@sentry/types';
import { LazyLoadedIntegration } from './lazy';
interface MysqlConnection {
    prototype: {
        connect: () => void;
    };
    createQuery: () => void;
}
/** Tracing integration for node-mysql package */
export declare class Mysql implements LazyLoadedIntegration<MysqlConnection> {
    /**
     * @inheritDoc
     */
    static id: string;
    /**
     * @inheritDoc
     */
    name: string;
    private _module?;
    constructor();
    /** @inheritdoc */
    loadDependency(): MysqlConnection | undefined;
    /**
     * @inheritDoc
     */
    setupOnce(_: (callback: EventProcessor) => void, getCurrentHub: () => Hub): void;
}
export {};
//# sourceMappingURL=mysql.d.ts.map
