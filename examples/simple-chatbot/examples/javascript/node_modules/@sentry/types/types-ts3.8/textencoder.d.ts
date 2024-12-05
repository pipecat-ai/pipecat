/**
 * Vendored type from TS 3.8 `typescript/lib/lib.dom.d.ts`.
 *
 * Type is vendored in so that users don't have to opt-in to DOM types.
 */
export interface TextEncoderCommon {
    /**
     * Returns "utf-8".
     */
    readonly encoding: string;
}
export interface TextEncoderInternal extends TextEncoderCommon {
    encode(input?: string): Uint8Array;
}
//# sourceMappingURL=textencoder.d.ts.map
