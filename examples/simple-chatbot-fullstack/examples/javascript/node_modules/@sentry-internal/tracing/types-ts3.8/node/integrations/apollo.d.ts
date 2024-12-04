import { Hub } from '@sentry/core';
import { EventProcessor } from '@sentry/types';
import { LazyLoadedIntegration } from './lazy';
interface ApolloOptions {
    useNestjs?: boolean;
}
type ApolloResolverGroup = {
    [key: string]: () => unknown;
};
type ApolloModelResolvers = {
    [key: string]: ApolloResolverGroup;
};
type GraphQLModule = {
    GraphQLFactory: {
        prototype: {
            create: (resolvers: ApolloModelResolvers[]) => unknown;
        };
    };
};
type ApolloModule = {
    ApolloServerBase: {
        prototype: {
            constructSchema: (config: unknown) => unknown;
        };
    };
};
/** Tracing integration for Apollo */
export declare class Apollo implements LazyLoadedIntegration<GraphQLModule & ApolloModule> {
    /**
     * @inheritDoc
     */
    static id: string;
    /**
     * @inheritDoc
     */
    name: string;
    private readonly _useNest;
    private _module?;
    /**
     * @inheritDoc
     */
    constructor(options?: ApolloOptions);
    /** @inheritdoc */
    loadDependency(): (GraphQLModule & ApolloModule) | undefined;
    /**
     * @inheritDoc
     */
    setupOnce(_: (callback: EventProcessor) => void, getCurrentHub: () => Hub): void;
}
export {};
//# sourceMappingURL=apollo.d.ts.map
