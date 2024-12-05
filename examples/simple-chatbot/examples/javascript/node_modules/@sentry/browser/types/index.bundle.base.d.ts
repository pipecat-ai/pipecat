export * from './exports';
import type { Integration } from '@sentry/types';
declare const INTEGRATIONS: Record<string, new (...args: any[]) => Integration>;
export { INTEGRATIONS as Integrations };
//# sourceMappingURL=index.bundle.base.d.ts.map