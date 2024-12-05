import { Client, Integration, IntegrationClass } from '@sentry/types';
type GlobalHandlersIntegrationsOptionKeys = 'onerror' | 'onunhandledrejection';
type GlobalHandlersIntegrations = Record<GlobalHandlersIntegrationsOptionKeys, boolean>;
export declare const globalHandlersIntegration: (options?: Partial<GlobalHandlersIntegrations> | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * Global handlers.
 * @deprecated Use `globalHandlersIntegration()` instead.
 */
export declare const GlobalHandlers: IntegrationClass<Integration & {
    setup: (client: Client) => void;
}> & (new (options?: Partial<GlobalHandlersIntegrations>) => Integration);
export {};
//# sourceMappingURL=globalhandlers.d.ts.map
