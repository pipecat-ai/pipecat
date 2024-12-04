import { Client, Integration, IntegrationClass } from '@sentry/types';
interface CaptureConsoleOptions {
    levels?: string[];
}
export declare const captureConsoleIntegration: (options?: CaptureConsoleOptions | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * Send Console API calls as Sentry Events.
 * @deprecated Use `captureConsoleIntegration()` instead.
 */
export declare const CaptureConsole: IntegrationClass<Integration & {
    setup: (client: Client) => void;
}> & (new (options?: {
    levels?: string[];
}) => Integration);
export {};
//# sourceMappingURL=captureconsole.d.ts.map
