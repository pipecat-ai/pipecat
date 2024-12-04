import type { FeedbackInternalOptions } from '../types';
type CreateShadowHostParams = Pick<FeedbackInternalOptions, 'id' | 'colorScheme' | 'themeDark' | 'themeLight'>;
/**
 * Creates shadow host
 */
export declare function createShadowHost({ id, colorScheme, themeDark, themeLight }: CreateShadowHostParams): {
    shadow: ShadowRoot;
    host: HTMLDivElement;
};
export {};
//# sourceMappingURL=createShadowHost.d.ts.map