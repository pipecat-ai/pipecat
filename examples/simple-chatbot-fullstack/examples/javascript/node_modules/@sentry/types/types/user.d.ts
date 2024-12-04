/**
 * An interface describing a user of an application or a handled request.
 */
export interface User {
    [key: string]: any;
    id?: string | number;
    ip_address?: string;
    email?: string;
    username?: string;
    /**
     * @deprecated Functonality for segment has been removed. Use a custom tag or context instead to capture this information.
     */
    segment?: string;
}
export interface UserFeedback {
    event_id: string;
    email: User['email'];
    name: string;
    comments: string;
}
//# sourceMappingURL=user.d.ts.map