import { Breadcrumb, Scope } from '@sentry/types';
import { ReplayContainer } from '../types';
import { ReplayFrame } from '../types/replayFrame';
type BreadcrumbWithCategory = Required<Pick<Breadcrumb, 'category'>>;
export declare const handleScopeListener: (replay: ReplayContainer) => (scope: Scope) => void;
/**
 * An event handler to handle scope changes.
 */
export declare function handleScope(scope: Scope): Breadcrumb | null;
/** exported for tests only */
export declare function normalizeConsoleBreadcrumb(breadcrumb: Pick<Breadcrumb, Exclude<keyof Breadcrumb, 'category'>> & BreadcrumbWithCategory): ReplayFrame;
export {};
//# sourceMappingURL=handleScope.d.ts.map
