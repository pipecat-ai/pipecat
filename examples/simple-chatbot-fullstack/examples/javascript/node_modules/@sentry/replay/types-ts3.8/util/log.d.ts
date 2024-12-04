/**
 * Log a message in debug mode, and add a breadcrumb when _experiment.traceInternals is enabled.
 */
export declare function logInfo(message: string, shouldAddBreadcrumb?: boolean): void;
/**
 * Log a message, and add a breadcrumb in the next tick.
 * This is necessary when the breadcrumb may be added before the replay is initialized.
 */
export declare function logInfoNextTick(message: string, shouldAddBreadcrumb?: boolean): void;
//# sourceMappingURL=log.d.ts.map
