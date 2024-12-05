import type { Event } from './event';
export interface FeedbackContext extends Record<string, unknown> {
    message: string;
    contact_email?: string;
    name?: string;
    replay_id?: string;
    url?: string;
}
/**
 * NOTE: These types are still considered Alpha and subject to change.
 * @hidden
 */
export interface FeedbackEvent extends Event {
    type: 'feedback';
    contexts: Event['contexts'] & {
        feedback: FeedbackContext;
    };
}
//# sourceMappingURL=feedback.d.ts.map