import type { Severity, SeverityLevel } from './severity';
/** JSDoc */
export interface Breadcrumb {
    type?: string;
    level?: Severity | SeverityLevel;
    event_id?: string;
    category?: string;
    message?: string;
    data?: {
        [key: string]: any;
    };
    timestamp?: number;
}
/** JSDoc */
export interface BreadcrumbHint {
    [key: string]: any;
}
export interface FetchBreadcrumbData {
    method: string;
    url: string;
    status_code?: number;
    request_body_size?: number;
    response_body_size?: number;
}
export interface XhrBreadcrumbData {
    method?: string;
    url?: string;
    status_code?: number;
    request_body_size?: number;
    response_body_size?: number;
}
export interface FetchBreadcrumbHint {
    input: any[];
    data?: unknown;
    response?: unknown;
    startTimestamp: number;
    endTimestamp: number;
}
export interface XhrBreadcrumbHint {
    xhr: unknown;
    input: unknown;
    startTimestamp: number;
    endTimestamp: number;
}
//# sourceMappingURL=breadcrumb.d.ts.map