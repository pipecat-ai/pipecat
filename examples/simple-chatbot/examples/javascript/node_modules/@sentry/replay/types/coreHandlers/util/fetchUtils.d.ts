import type { Breadcrumb, FetchBreadcrumbData, TextEncoderInternal } from '@sentry/types';
import type { FetchHint, ReplayContainer, ReplayNetworkOptions, ReplayNetworkRequestOrResponse } from '../../types';
/**
 * Capture a fetch breadcrumb to a replay.
 * This adds additional data (where approriate).
 */
export declare function captureFetchBreadcrumbToReplay(breadcrumb: Breadcrumb & {
    data: FetchBreadcrumbData;
}, hint: Partial<FetchHint>, options: ReplayNetworkOptions & {
    textEncoder: TextEncoderInternal;
    replay: ReplayContainer;
}): Promise<void>;
/**
 * Enrich a breadcrumb with additional data.
 * This has to be sync & mutate the given breadcrumb,
 * as the breadcrumb is afterwards consumed by other handlers.
 */
export declare function enrichFetchBreadcrumb(breadcrumb: Breadcrumb & {
    data: FetchBreadcrumbData;
}, hint: Partial<FetchHint>, options: {
    textEncoder: TextEncoderInternal;
}): void;
/** Exported only for tests. */
export declare function _getResponseInfo(captureDetails: boolean, { networkCaptureBodies, textEncoder, networkResponseHeaders, }: Pick<ReplayNetworkOptions, 'networkCaptureBodies' | 'networkResponseHeaders'> & {
    textEncoder: TextEncoderInternal;
}, response: Response | undefined, responseBodySize?: number): Promise<ReplayNetworkRequestOrResponse | undefined>;
//# sourceMappingURL=fetchUtils.d.ts.map