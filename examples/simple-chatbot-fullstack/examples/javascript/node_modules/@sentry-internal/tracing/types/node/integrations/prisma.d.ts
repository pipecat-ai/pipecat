import type { Integration } from '@sentry/types';
/** Tracing integration for @prisma/client package */
export declare class Prisma implements Integration {
    /**
     * @inheritDoc
     */
    static id: string;
    /**
     * @inheritDoc
     */
    name: string;
    /**
     * @inheritDoc
     */
    constructor(options?: {
        client?: unknown;
    });
    /**
     * @inheritDoc
     */
    setupOnce(): void;
}
//# sourceMappingURL=prisma.d.ts.map