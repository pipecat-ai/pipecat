import type { Client, Integration, IntegrationClass } from '@sentry/types';
interface BreadcrumbsOptions {
    console: boolean;
    dom: boolean | {
        serializeAttribute?: string | string[];
        maxStringLength?: number;
    };
    fetch: boolean;
    history: boolean;
    sentry: boolean;
    xhr: boolean;
}
export declare const breadcrumbsIntegration: (options?: Partial<BreadcrumbsOptions> | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * Default Breadcrumbs instrumentations
 *
 * @deprecated Use `breadcrumbsIntegration()` instead.
 */
export declare const Breadcrumbs: IntegrationClass<Integration & {
    setup: (client: Client) => void;
}> & (new (options?: Partial<{
    console: boolean;
    dom: boolean | {
        serializeAttribute?: string | string[];
        maxStringLength?: number;
    };
    fetch: boolean;
    history: boolean;
    sentry: boolean;
    xhr: boolean;
}>) => Integration);
export {};
//# sourceMappingURL=breadcrumbs.d.ts.map