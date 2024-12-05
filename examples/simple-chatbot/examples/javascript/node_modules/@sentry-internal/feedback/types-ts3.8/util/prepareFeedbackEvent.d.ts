import { Scope } from '@sentry/core';
import { Client, FeedbackEvent } from '@sentry/types';
interface PrepareFeedbackEventParams {
    client: Client;
    event: FeedbackEvent;
    scope: Scope;
}
/**
 * Prepare a feedback event & enrich it with the SDK metadata.
 */
export declare function prepareFeedbackEvent({ client, scope, event, }: PrepareFeedbackEventParams): Promise<FeedbackEvent | null>;
export {};
//# sourceMappingURL=prepareFeedbackEvent.d.ts.map
