import type { Integration } from '@sentry/types';
export type UserIntegrationsFunction = (integrations: Integration[]) => Integration[];
export type UserIntegrations = Integration[] | UserIntegrationsFunction;
export type IntegrationWithExclusionOption = Integration & {
    /**
     * Allow the user to exclude this integration by not returning it from a function provided as the `integrations` option
     * in `Sentry.init()`. Meant to be used with default integrations, the idea being that if a user has actively filtered
     * an integration out, we should be able to respect that choice if we wish.
     */
    allowExclusionByUser?: boolean;
};
type ForcedIntegrationOptions = {
    [keyPath: string]: unknown;
};
/**
 * Enforces inclusion of a given integration with specified options in an integration array originally determined by the
 * user, by either including the given default instance or by patching an existing user instance with the given options.
 *
 * Ideally this would happen when integrations are set up, but there isn't currently a mechanism there for merging
 * options from a default integration instance with those from a user-provided instance of the same integration, only
 * for allowing the user to override a default instance entirely. (TODO: Fix that.)
 *
 * @param defaultIntegrationInstance An instance of the integration with the correct options already set
 * @param userIntegrations Integrations defined by the user.
 * @param forcedOptions Options with which to patch an existing user-derived instance on the integration.
 * @returns A final integrations array.
 *
 * @deprecated This will be removed in v8.
 */
export declare function addOrUpdateIntegration<T extends UserIntegrations>(defaultIntegrationInstance: Integration, userIntegrations: T, forcedOptions?: ForcedIntegrationOptions): T;
export {};
//# sourceMappingURL=userIntegrations.d.ts.map