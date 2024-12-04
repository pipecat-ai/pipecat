import { Hub } from '@sentry/core';
import { EventProcessor } from '@sentry/types';
import { LazyLoadedIntegration } from './lazy';
type Operation = (typeof OPERATIONS)[number];
declare const OPERATIONS: readonly [
    "aggregate",
    "bulkWrite",
    "countDocuments",
    "createIndex",
    "createIndexes",
    "deleteMany",
    "deleteOne",
    "distinct",
    "drop",
    "dropIndex",
    "dropIndexes",
    "estimatedDocumentCount",
    "find",
    "findOne",
    "findOneAndDelete",
    "findOneAndReplace",
    "findOneAndUpdate",
    "indexes",
    "indexExists",
    "indexInformation",
    "initializeOrderedBulkOp",
    "insertMany",
    "insertOne",
    "isCapped",
    "mapReduce",
    "options",
    "parallelCollectionScan",
    "rename",
    "replaceOne",
    "stats",
    "updateMany",
    "updateOne"
];
interface MongoCollection {
    collectionName: string;
    dbName: string;
    namespace: string;
    prototype: {
        [operation in Operation]: (...args: unknown[]) => unknown;
    };
}
interface MongoOptions {
    operations?: Operation[];
    describeOperations?: boolean | Operation[];
    useMongoose?: boolean;
}
type MongoModule = {
    Collection: MongoCollection;
};
/** Tracing integration for mongo package */
export declare class Mongo implements LazyLoadedIntegration<MongoModule> {
    /**
     * @inheritDoc
     */
    static id: string;
    /**
     * @inheritDoc
     */
    name: string;
    private _operations;
    private _describeOperations?;
    private _useMongoose;
    private _module?;
    /**
     * @inheritDoc
     */
    constructor(options?: MongoOptions);
    /** @inheritdoc */
    loadDependency(): MongoModule | undefined;
    /**
     * @inheritDoc
     */
    setupOnce(_: (callback: EventProcessor) => void, getCurrentHub: () => Hub): void;
    /**
     * Patches original collection methods
     */
    private _instrumentOperations;
    /**
     * Patches original collection to utilize our tracing functionality
     */
    private _patchOperation;
    /**
     * Form a SpanContext based on the user input to a given operation.
     */
    private _getSpanContextFromOperationArguments;
}
export {};
//# sourceMappingURL=mongo.d.ts.map
