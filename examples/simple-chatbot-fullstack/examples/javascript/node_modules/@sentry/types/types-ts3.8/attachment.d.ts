export type AttachmentType = 'event.attachment' | 'event.minidump' | 'event.applecrashreport' | 'unreal.context' | 'unreal.logs';
/**
 * An attachment to an event. This is used to upload arbitrary data to Sentry.
 *
 * Please take care to not add sensitive information in attachments.
 *
 * https://develop.sentry.dev/sdk/envelopes/#attachment
 */
export interface Attachment {
    data: string | Uint8Array;
    filename: string;
    contentType?: string;
    /**
     * The type of the attachment. Defaults to `event.attachment` if not specified.
     */
    attachmentType?: AttachmentType;
}
//# sourceMappingURL=attachment.d.ts.map
