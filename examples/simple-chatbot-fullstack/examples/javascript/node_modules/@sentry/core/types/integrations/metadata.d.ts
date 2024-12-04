import type { Client, Event, EventHint, Integration, IntegrationClass } from '@sentry/types';
export declare const moduleMetadataIntegration: () => import("@sentry/types").IntegrationFnResult;
/**
 * Adds module metadata to stack frames.
 *
 * Metadata can be injected by the Sentry bundler plugins using the `_experiments.moduleMetadata` config option.
 *
 * When this integration is added, the metadata passed to the bundler plugin is added to the stack frames of all events
 * under the `module_metadata` property. This can be used to help in tagging or routing of events from different teams
 * our sources
 *
 * @deprecated Use `moduleMetadataIntegration()` instead.
 */
export declare const ModuleMetadata: IntegrationClass<Integration & {
    setup: (client: Client) => void;
    processEvent: (event: Event, hint: EventHint, client: Client) => Event;
}>;
//# sourceMappingURL=metadata.d.ts.map