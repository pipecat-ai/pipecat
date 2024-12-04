import type { Breadcrumb, Scope } from '@sentry/types';
import type { ReplayContainer } from '../types';
import type { ReplayFrame } from '../types/replayFrame';
type BreadcrumbWithCategory = Required<Pick<Breadcrumb, 'category'>>;
export declare const handleScopeListener: (replay: ReplayContainer) => (scope: Scope) => void;
/**
 * An event handler to handle scope changes.
 */
export declare function handleScope(scope: Scope): Breadcrumb | null;
/** exported for tests only */
export declare function normalizeConsoleBreadcrumb(breadcrumb: Omit<Breadcrumb, 'category'> & BreadcrumbWithCategory): ReplayFrame;
export {};
//# sourceMappingURL=handleScope.d.ts.map