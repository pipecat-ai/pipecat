import { SendFeedbackOptions } from './types';
import { sendFeedbackRequest } from './util/sendFeedbackRequest';
interface SendFeedbackParams {
    message: string;
    name?: string;
    email?: string;
    url?: string;
    source?: string;
}
/**
 * Public API to send a Feedback item to Sentry
 */
export declare function sendFeedback({ name, email, message, source, url }: SendFeedbackParams, options?: SendFeedbackOptions): ReturnType<typeof sendFeedbackRequest>;
export {};
//# sourceMappingURL=sendFeedback.d.ts.map
