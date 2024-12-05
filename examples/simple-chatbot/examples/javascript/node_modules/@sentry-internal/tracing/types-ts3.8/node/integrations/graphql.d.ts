import { Hub } from '@sentry/core';
import { EventProcessor } from '@sentry/types';
import { LazyLoadedIntegration } from './lazy';
type GraphQLModule = {
    [method: string]: (...args: unknown[]) => unknown;
};
/** Tracing integration for graphql package */
export declare class GraphQL implements LazyLoadedIntegration<GraphQLModule> {
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
    loadDependency(): GraphQLModule | undefined;
    /**
     * @inheritDoc
     */
    setupOnce(_: (callback: EventProcessor) => void, getCurrentHub: () => Hub): void;
}
export {};
//# sourceMappingURL=graphql.d.ts.map
