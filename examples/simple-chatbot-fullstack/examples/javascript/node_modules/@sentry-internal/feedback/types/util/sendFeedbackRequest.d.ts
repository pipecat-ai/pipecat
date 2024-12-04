import type { TransportMakeRequestResponse } from '@sentry/types';
import type { SendFeedbackData, SendFeedbackOptions } from '../types';
/**
 * Send feedback using transport
 */
export declare function sendFeedbackRequest({ feedback: { message, email, name, source, url } }: SendFeedbackData, { includeReplay }?: SendFeedbackOptions): Promise<void | TransportMakeRequestResponse>;
//# sourceMappingURL=sendFeedbackRequest.d.ts.map