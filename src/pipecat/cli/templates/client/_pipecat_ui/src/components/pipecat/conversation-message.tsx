"use client";

import type {
  AggregationMetadata,
  BotOutputText,
  ConversationMessage,
  ConversationMessagePart,
  FunctionCallData,
  FunctionCallRenderer,
} from "@pipecat-ai/client-react";
import { isMessageEmpty } from "@pipecat-ai/client-react";
import {
  CheckIcon,
  ChevronRightIcon,
  LoaderCircleIcon,
  XIcon,
} from "lucide-react";
import { Fragment, useEffect, useState } from "react";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

/** Controls how bot message text is rendered. */
export type TextRenderMode = "karaoke" | "captions" | "instant";

export type BotOutputRenderer = (
  content: string,
  metadata: { spoken: string; unspoken: string },
) => React.ReactNode;

const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: "numeric",
  minute: "numeric",
  second: "numeric",
});

/** Animated "…" typing indicator. */
export function Thinking({
  className,
  interval = 500,
  maxDots = 3,
}: {
  className?: string;
  interval?: number;
  maxDots?: number;
}) {
  const [dots, setDots] = useState(1);

  useEffect(() => {
    const i = setInterval(() => {
      setDots((prev) => (prev % maxDots) + 1);
    }, interval);
    return () => clearInterval(i);
  }, [interval, maxDots]);

  return (
    <span className={className} aria-label="Thinking">
      {".".repeat(dots)}
    </span>
  );
}

export interface MessageRoleProps {
  role: ConversationMessage["role"];
  assistantLabel?: string;
  clientLabel?: string;
  systemLabel?: string;
  functionCallLabel?: string;
  className?: string;
}

/** Role label for a conversation message, colored by participant. */
export function MessageRole({
  role,
  assistantLabel = "assistant",
  clientLabel = "user",
  systemLabel = "system",
  functionCallLabel = "function call",
  className,
}: MessageRoleProps) {
  const labels: Record<string, string> = {
    user: clientLabel,
    assistant: assistantLabel,
    system: systemLabel,
    function_call: functionCallLabel,
  };

  return (
    <div
      data-slot="message-role"
      className={cn(
        "w-max font-mono text-xs leading-6 font-semibold",
        {
          "text-client-foreground": role === "user",
          "text-agent-foreground": role === "assistant",
          "text-muted-foreground":
            role === "system" || role === "function_call",
        },
        className,
      )}
    >
      {labels[role] || role}
    </div>
  );
}

function StatusIcon({
  status,
  cancelled,
}: {
  status: FunctionCallData["status"];
  cancelled?: boolean;
}) {
  if (status === "completed" && cancelled) {
    return <XIcon className="text-destructive size-3.5" />;
  }
  switch (status) {
    case "started":
    case "in_progress":
      return <LoaderCircleIcon className="size-3.5 animate-spin" />;
    case "completed":
      return <CheckIcon className="text-active size-3.5" />;
  }
}

export interface FunctionCallContentProps {
  functionCall: FunctionCallData;
  functionCallLabel?: string;
  /** Replaces the default rendering entirely. */
  functionCallRenderer?: FunctionCallRenderer;
  className?: string;
}

/** Collapsible display of a function call with its args and result. */
export function FunctionCallContent({
  functionCall,
  functionCallLabel = "Function call",
  functionCallRenderer,
  className,
}: FunctionCallContentProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (functionCallRenderer) {
    return <>{functionCallRenderer(functionCall)}</>;
  }

  const hasDetails =
    (functionCall.args && Object.keys(functionCall.args).length > 0) ||
    functionCall.result !== undefined;

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className={cn("flex flex-col gap-1", className)}>
        <CollapsibleTrigger
          disabled={!hasDetails}
          className={cn(
            "text-muted-foreground flex items-center gap-2 font-mono text-xs transition-colors select-none",
            hasDetails
              ? "hover:text-foreground cursor-pointer"
              : "cursor-default",
          )}
        >
          {hasDetails && (
            <ChevronRightIcon
              className={cn(
                "size-3.5 transition-transform duration-200",
                isOpen && "rotate-90",
              )}
            />
          )}
          <StatusIcon
            status={functionCall.status}
            cancelled={functionCall.cancelled}
          />
          <span className="font-semibold">{functionCallLabel}</span>
          {functionCall.function_name && (
            <span className="text-muted-foreground">
              ({functionCall.function_name})
            </span>
          )}
        </CollapsibleTrigger>

        {hasDetails && (
          <CollapsibleContent>
            <div className="border-muted mt-1 ml-3.5 flex flex-col gap-2 border-l-2 pl-3 font-mono text-xs">
              {functionCall.args &&
                Object.keys(functionCall.args).length > 0 && (
                  <div>
                    <div className="text-muted-foreground mb-1 font-semibold">
                      Arguments
                    </div>
                    <pre className="bg-muted/50 overflow-x-auto rounded p-2 break-all whitespace-pre-wrap">
                      {JSON.stringify(functionCall.args, null, 2)}
                    </pre>
                  </div>
                )}
              {functionCall.result !== undefined && (
                <div>
                  <div className="text-muted-foreground mb-1 font-semibold">
                    Result
                  </div>
                  <pre className="bg-muted/50 overflow-x-auto rounded p-2 break-all whitespace-pre-wrap">
                    {typeof functionCall.result === "string"
                      ? functionCall.result
                      : JSON.stringify(functionCall.result, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </CollapsibleContent>
        )}
      </div>
    </Collapsible>
  );
}

function isBotOutputText(
  part: ConversationMessagePart,
): part is ConversationMessagePart & { text: BotOutputText } {
  const text = part.text;
  return (
    text !== null &&
    typeof text === "object" &&
    "spoken" in text &&
    "unspoken" in text
  );
}

function renderBotOutput(
  spoken: string,
  unspoken: string,
  aggregatedBy?: string,
  customRenderer?: BotOutputRenderer,
  metadata?: AggregationMetadata,
  textRenderMode?: TextRenderMode,
): React.ReactNode {
  if (aggregatedBy && customRenderer) {
    return customRenderer(spoken + unspoken, { spoken, unspoken });
  }

  const displayMode = metadata?.displayMode || "inline";
  const Wrapper = displayMode === "block" ? "div" : "span";

  if (textRenderMode === "instant") {
    return (
      <Wrapper>
        {spoken}
        {unspoken}
      </Wrapper>
    );
  }

  return (
    <Wrapper>
      {spoken}
      {unspoken && <span className="text-muted-foreground">{unspoken}</span>}
    </Wrapper>
  );
}

function renderPartContent(
  part: ConversationMessagePart,
  botOutputRenderers?: Record<string, BotOutputRenderer>,
  aggregationMetadata?: Record<string, AggregationMetadata>,
  textRenderMode?: TextRenderMode,
): React.ReactNode {
  if (!part.text) return null;
  if (isBotOutputText(part)) {
    const text = part.text as BotOutputText;
    const customRenderer = part.aggregatedBy
      ? botOutputRenderers?.[part.aggregatedBy]
      : undefined;
    const metadata = part.aggregatedBy
      ? aggregationMetadata?.[part.aggregatedBy]
      : undefined;
    return renderBotOutput(
      text.spoken,
      text.unspoken,
      part.aggregatedBy,
      customRenderer,
      metadata,
      textRenderMode,
    );
  }
  return part.text as React.ReactNode;
}

export interface MessageContentProps {
  message: ConversationMessage;
  botOutputRenderers?: Record<string, BotOutputRenderer>;
  aggregationMetadata?: Record<string, AggregationMetadata>;
  textRenderMode?: TextRenderMode;
  className?: string;
}

/** Message body: grouped inline/block parts, thinking dots, timestamp. */
export function MessageContent({
  message,
  botOutputRenderers,
  aggregationMetadata,
  textRenderMode,
  className,
}: MessageContentProps) {
  const parts = Array.isArray(message.parts) ? message.parts : [];

  // Group parts by display mode: inline parts flow together, block parts
  // each get their own line.
  const groupedParts: Array<{
    type: "inline" | "block";
    parts: ConversationMessagePart[];
  }> = [];
  let currentInlineGroup: ConversationMessagePart[] = [];

  for (const part of parts) {
    const metadata = part.aggregatedBy
      ? aggregationMetadata?.[part.aggregatedBy]
      : undefined;
    const displayMode = part.displayMode ?? metadata?.displayMode ?? "inline";

    if (displayMode === "block") {
      if (currentInlineGroup.length > 0) {
        groupedParts.push({ type: "inline", parts: currentInlineGroup });
        currentInlineGroup = [];
      }
      groupedParts.push({ type: "block", parts: [part] });
    } else {
      currentInlineGroup.push(part);
    }
  }
  if (currentInlineGroup.length > 0) {
    groupedParts.push({ type: "inline", parts: currentInlineGroup });
  }

  return (
    <div
      data-slot="message-content"
      className={cn("flex flex-col gap-2", className)}
    >
      {groupedParts.map((group, groupIdx) =>
        group.type === "inline" ? (
          <div key={groupIdx} className="inline-block">
            {group.parts.map((part, partIdx) => {
              const content = renderPartContent(
                part,
                botOutputRenderers,
                aggregationMetadata,
                textRenderMode,
              );
              const shouldAddSpace = partIdx > 0 && !isBotOutputText(part);
              return (
                <Fragment key={partIdx}>
                  {shouldAddSpace && " "}
                  {content}
                </Fragment>
              );
            })}
          </div>
        ) : (
          <Fragment key={groupIdx}>
            {group.parts.map((part, partIdx) => (
              <Fragment key={partIdx}>
                {renderPartContent(
                  part,
                  botOutputRenderers,
                  aggregationMetadata,
                  textRenderMode,
                )}
              </Fragment>
            ))}
          </Fragment>
        ),
      )}
      {isMessageEmpty(message) ? <Thinking /> : null}
      <div className="text-muted-foreground mb-1 self-end text-xs">
        {timeFormatter.format(new Date(message.createdAt))}
      </div>
    </div>
  );
}

export interface ConversationMessageItemProps {
  message: ConversationMessage;
  assistantLabel?: string;
  clientLabel?: string;
  systemLabel?: string;
  functionCallLabel?: string;
  functionCallRenderer?: FunctionCallRenderer;
  botOutputRenderers?: Record<string, BotOutputRenderer>;
  aggregationMetadata?: Record<string, AggregationMetadata>;
  textRenderMode?: TextRenderMode;
  className?: string;
}

/** One conversation row: role label plus content (or a function call). */
export function ConversationMessageItem({
  message,
  assistantLabel,
  clientLabel,
  systemLabel,
  functionCallLabel,
  functionCallRenderer,
  botOutputRenderers,
  aggregationMetadata,
  textRenderMode,
  className,
}: ConversationMessageItemProps) {
  if (message.role === "function_call" && message.functionCall) {
    return (
      <div className={cn("flex flex-col gap-2", className)}>
        <FunctionCallContent
          functionCall={message.functionCall}
          functionCallLabel={functionCallLabel}
          functionCallRenderer={functionCallRenderer}
        />
        <div className="text-muted-foreground mb-1 self-end text-xs">
          {timeFormatter.format(new Date(message.createdAt))}
        </div>
      </div>
    );
  }

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <MessageRole
        role={message.role}
        assistantLabel={assistantLabel}
        clientLabel={clientLabel}
        systemLabel={systemLabel}
        functionCallLabel={functionCallLabel}
      />
      <MessageContent
        message={message}
        botOutputRenderers={botOutputRenderers}
        aggregationMetadata={aggregationMetadata}
        textRenderMode={textRenderMode}
      />
    </div>
  );
}
