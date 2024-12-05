/**
 * Helper function to create an element. Could be used as a JSX factory
 * (i.e. React-like syntax).
 */
export declare function createElement<K extends keyof HTMLElementTagNameMap>(tagName: K, attributes: {
    [key: string]: string | boolean | EventListenerOrEventListenerObject;
} | null, ...children: any): HTMLElementTagNameMap[K];
//# sourceMappingURL=createElement.d.ts.map
