import type { StackLineParserFn } from '@sentry/types';
export type GetModuleFn = (filename: string | undefined) => string | undefined;
/**
 * Does this filename look like it's part of the app code?
 */
export declare function filenameIsInApp(filename: string, isNative?: boolean): boolean;
/** Node Stack line parser */
export declare function node(getModule?: GetModuleFn): StackLineParserFn;
//# sourceMappingURL=node-stack-trace.d.ts.map