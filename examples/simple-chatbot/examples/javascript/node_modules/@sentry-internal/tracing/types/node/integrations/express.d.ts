import type { Hub, Integration, PolymorphicRequest } from '@sentry/types';
type Method = 'all' | 'get' | 'post' | 'put' | 'delete' | 'patch' | 'options' | 'head' | 'checkout' | 'copy' | 'lock' | 'merge' | 'mkactivity' | 'mkcol' | 'move' | 'm-search' | 'notify' | 'purge' | 'report' | 'search' | 'subscribe' | 'trace' | 'unlock' | 'unsubscribe' | 'use';
type Router = {
    [method in Method]: (...args: any) => any;
};
type PatchedRequest = PolymorphicRequest & {
    _reconstructedRoute?: string;
    _hasParameters?: boolean;
};
type Layer = {
    match: (path: string) => boolean;
    handle_request: (req: PatchedRequest, res: ExpressResponse, next: () => void) => void;
    route?: {
        path: RouteType | RouteType[];
    };
    path?: string;
    regexp?: RegExp;
    keys?: {
        name: string | number;
        offset: number;
        optional: boolean;
    }[];
};
type RouteType = string | RegExp;
interface ExpressResponse {
    once(name: string, callback: () => void): void;
}
/**
 * Express integration
 *
 * Provides an request and error handler for Express framework as well as tracing capabilities
 */
export declare class Express implements Integration {
    /**
     * @inheritDoc
     */
    static id: string;
    /**
     * @inheritDoc
     */
    name: string;
    /**
     * Express App instance
     */
    private readonly _router?;
    private readonly _methods?;
    /**
     * @inheritDoc
     */
    constructor(options?: {
        app?: Router;
        router?: Router;
        methods?: Method[];
    });
    /**
     * @inheritDoc
     */
    setupOnce(_: unknown, getCurrentHub: () => Hub): void;
}
/**
 * Recreate layer.route.path from layer.regexp and layer.keys.
 * Works until express.js used package path-to-regexp@0.1.7
 * or until layer.keys contain offset attribute
 *
 * @param layer the layer to extract the stringified route from
 *
 * @returns string in layer.route.path structure 'router/:pathParam' or undefined
 */
export declare const extractOriginalRoute: (path?: Layer['path'], regexp?: Layer['regexp'], keys?: Layer['keys']) => string | undefined;
/**
 * remove duplicate segment contain in layerPath against reconstructedRoute,
 * and return only unique segment that can be added into reconstructedRoute
 */
export declare function preventDuplicateSegments(originalUrl?: string, reconstructedRoute?: string, layerPath?: string): string | undefined;
export {};
//# sourceMappingURL=express.d.ts.map