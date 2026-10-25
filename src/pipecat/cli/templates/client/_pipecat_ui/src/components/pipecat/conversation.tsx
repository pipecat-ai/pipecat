"use client";

import type { ConversationMessage } from "@pipecat-ai/client-react";
import {
  useConversationContext,
  usePipecatClientTransportState,
  usePipecatConversation,
} from "@pipecat-ai/client-react";
import { useCallback, useEffect, useMemo, useRef } from "react";

import {
  ConversationMessageItem,
  type ConversationMessageItemProps,
  type TextRenderMode,
} from "@/components/pipecat/conversation-message";
import { cn } from "@/lib/utils";

export interface ConversationViewProps extends Omit<
  ConversationMessageItemProps,
  "message" | "className"
> {
  /** Messages to render. */
  messages?: ConversationMessage[];
  /** Disable automatic scrolling when new messages arrive. */
  noAutoscroll?: boolean;
  /** Newest messages first; auto-scroll targets the top. */
  reverseOrder?: boolean;
  /** Rendered when there are no messages. */
  empty?: React.ReactNode;
  className?: string;
  /** Classes for each message row. */
  messageClassName?: string;
}

/**
 * Scrolling conversation list. Auto-scrolls to the newest message unless
 * the user has scrolled away. Fully props-driven; pair with the connected
 * Conversation for Pipecat wiring.
 */
export function ConversationView({
  messages = [],
  noAutoscroll = false,
  reverseOrder = false,
  empty,
  className,
  messageClassName,
  ...messageProps
}: ConversationViewProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const isScrolledToEdge = useRef(true);

  const maybeScrollToEdge = useCallback(() => {
    if (!scrollRef.current) return;
    if (isScrolledToEdge.current) {
      scrollRef.current.scrollTo({
        top: reverseOrder ? 0 : scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [reverseOrder]);

  const updateScrollState = useCallback(() => {
    if (!scrollRef.current || noAutoscroll) return;
    if (reverseOrder) {
      isScrolledToEdge.current = scrollRef.current.scrollTop <= 1;
    } else {
      isScrolledToEdge.current =
        Math.ceil(
          scrollRef.current.scrollHeight - scrollRef.current.scrollTop,
        ) <= Math.ceil(scrollRef.current.clientHeight);
    }
  }, [noAutoscroll, reverseOrder]);

  useEffect(() => {
    if (noAutoscroll) return;
    if (messages.length === 0) isScrolledToEdge.current = true;
    maybeScrollToEdge();
  }, [messages, maybeScrollToEdge, noAutoscroll]);

  if (messages.length === 0) {
    return (
      <div
        data-slot="conversation"
        className={cn("relative flex h-full flex-col", className)}
      >
        <div className="flex flex-1 items-center justify-center">
          {empty ?? (
            <div className="text-muted-foreground text-sm">
              Waiting for messages…
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div
      data-slot="conversation"
      className={cn("relative flex h-full flex-col", className)}
    >
      <div
        ref={scrollRef}
        onScroll={updateScrollState}
        className="relative flex-1 overflow-y-auto p-4"
      >
        <div className="flex flex-col gap-4">
          {(reverseOrder ? [...messages].reverse() : messages).map(
            (message, index) => (
              <ConversationMessageItem
                key={`${message.createdAt}-${index}`}
                message={message}
                className={messageClassName}
                {...messageProps}
              />
            ),
          )}
        </div>
      </div>
    </div>
  );
}

export interface ConversationProps extends Omit<
  ConversationViewProps,
  "messages" | "empty"
> {
  /** Hide function-call messages (they remain captured in the store). */
  noFunctionCalls?: boolean;
  /** Controls how bot message text is rendered. */
  textRenderMode?: TextRenderMode;
}

/**
 * Live conversation wired to the Pipecat client, with connection-aware
 * empty states. Must be rendered inside a PipecatClientProvider.
 */
export function Conversation({
  noFunctionCalls = false,
  textRenderMode,
  aggregationMetadata,
  ...props
}: ConversationProps) {
  const transportState = usePipecatClientTransportState();

  // "captions" mode only needs spoken text; the other modes need all data.
  const botOutputFilter = useMemo(() => {
    if (textRenderMode === "captions") return { unspoken: false };
    return undefined;
  }, [textRenderMode]);

  const { messages: allMessages } = usePipecatConversation({
    aggregationMetadata,
    botOutputFilter,
  });
  const { botOutputSupported } = useConversationContext();

  const messages = useMemo(
    () =>
      noFunctionCalls
        ? allMessages.filter((m) => m.role !== "function_call")
        : allMessages,
    [allMessages, noFunctionCalls],
  );

  const isConnecting =
    transportState === "authenticating" || transportState === "connecting";
  const isConnected =
    transportState === "connected" || transportState === "ready";

  let empty: React.ReactNode;
  if (isConnecting) {
    empty = (
      <div className="text-muted-foreground text-sm">Connecting to agent…</div>
    );
  } else if (!isConnected) {
    empty = (
      <div className="p-4 text-center">
        <div className="text-muted-foreground mb-2">Not connected to agent</div>
        <p className="text-muted-foreground max-w-md text-sm">
          Connect to an agent to see conversation messages in real-time.
        </p>
      </div>
    );
  } else if (botOutputSupported === false) {
    empty = (
      <div className="max-w-md p-4 text-center">
        <div className="text-destructive mb-2 font-medium">
          BotOutput events not supported
        </div>
        <p className="text-muted-foreground text-sm">
          This server does not support BotOutput events (requires RTVI 1.1.0+).
          Conversation messages cannot be displayed without them.
        </p>
      </div>
    );
  }

  return (
    <ConversationView
      messages={messages}
      empty={empty}
      textRenderMode={textRenderMode}
      aggregationMetadata={aggregationMetadata}
      {...props}
    />
  );
}
