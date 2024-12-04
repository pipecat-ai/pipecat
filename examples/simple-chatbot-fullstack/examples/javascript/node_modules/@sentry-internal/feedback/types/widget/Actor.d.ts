import type { FeedbackComponent, FeedbackInternalOptions } from '../types';
export interface ActorProps extends Pick<FeedbackInternalOptions, 'buttonLabel'> {
    onClick?: (e: MouseEvent) => void;
}
export interface ActorComponent extends FeedbackComponent<HTMLButtonElement> {
    /**
     * Shows the actor element
     */
    show: () => void;
    /**
     * Hides the actor element
     */
    hide: () => void;
}
/**
 *
 */
export declare function Actor({ buttonLabel, onClick }: ActorProps): ActorComponent;
//# sourceMappingURL=Actor.d.ts.map