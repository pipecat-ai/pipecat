# Installation
> `npm install --save @types/events`

# Summary
This package contains type definitions for events (https://github.com/Gozala/events).

# Details
Files were exported from https://github.com/DefinitelyTyped/DefinitelyTyped/tree/master/types/events.
## [index.d.ts](https://github.com/DefinitelyTyped/DefinitelyTyped/tree/master/types/events/index.d.ts)
````ts
export type Listener = (...args: any[]) => void;

export class EventEmitter {
    static listenerCount(emitter: EventEmitter, type: string | number): number;
    static defaultMaxListeners: number;

    eventNames(): Array<string | number>;
    setMaxListeners(n: number): this;
    getMaxListeners(): number;
    emit(type: string | number, ...args: any[]): boolean;
    addListener(type: string | number, listener: Listener): this;
    on(type: string | number, listener: Listener): this;
    once(type: string | number, listener: Listener): this;
    prependListener(type: string | number, listener: Listener): this;
    prependOnceListener(type: string | number, listener: Listener): this;
    removeListener(type: string | number, listener: Listener): this;
    off(type: string | number, listener: Listener): this;
    removeAllListeners(type?: string | number): this;
    listeners(type: string | number): Listener[];
    listenerCount(type: string | number): number;
    rawListeners(type: string | number): Listener[];
}

````

### Additional Details
 * Last updated: Mon, 06 Nov 2023 22:41:05 GMT
 * Dependencies: none

# Credits
These definitions were written by [Yasunori Ohoka](https://github.com/yasupeke), and [Shenwei Wang](https://github.com/weareoutman).
