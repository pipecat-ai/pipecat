import { Client, Hub } from '@sentry/types';
/**
 * Checks whether given url points to Sentry server
 * @param url url to verify
 *
 * TODO(v8): Remove Hub fallback type
 */
export declare function isSentryRequestUrl(url: string, hubOrClient: Hub | Client | undefined): boolean;
//# sourceMappingURL=isSentryRequestUrl.d.ts.map
