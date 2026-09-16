"use client";

import type { SendTextOptions, TransportState } from "@pipecat-ai/client-js";
import {
  useConversationContext,
  usePipecatClient,
  usePipecatClientTransportState,
} from "@pipecat-ai/client-react";
import { Loader2Icon, SendIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
  InputGroupTextarea,
} from "@/components/ui/input-group";

export interface TextInputViewProps {
  /** Called with the trimmed message when the user sends. */
  onSend?: (message: string) => Promise<void> | void;
  /** Disable the composer entirely. */
  disabled?: boolean;
  /** Use a textarea; Enter still sends, Shift+Enter inserts a newline. */
  multiline?: boolean;
  placeholder?: string;
  /** Accessible name for the input. */
  "aria-label"?: string;
  /** Content for the send button. Defaults to a send icon. */
  buttonContent?: React.ReactNode;
  /** Side the send button sits on in multiline mode. */
  buttonPosition?: "left" | "right";
  /** Overrides merged onto the send button (variant, size, …). */
  buttonProps?: Partial<React.ComponentProps<typeof InputGroupButton>>;
  /** Classes for the InputGroup root. */
  className?: string;
}

/** Enter sends unless composing with an IME. Successful sends clear the draft; failures preserve it. */
export function TextInputView({
  onSend,
  disabled = false,
  multiline = false,
  placeholder = "Type message…",
  "aria-label": ariaLabel = "Message",
  buttonContent,
  buttonPosition = "right",
  buttonProps,
  className,
}: TextInputViewProps) {
  const [message, setMessage] = useState("");
  const [isSending, setIsSending] = useState(false);
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null);

  // The input is disabled while a send is in flight, which drops focus.
  // Restore it once the send settles (success or failure) — after the
  // re-render that re-enables the input, hence the effect.
  const wasSending = useRef(false);
  useEffect(() => {
    if (wasSending.current && !isSending) inputRef.current?.focus();
    wasSending.current = isSending;
  }, [isSending]);

  const handleSend = useCallback(async () => {
    const text = message.trim();
    if (!text || isSending) return;
    setIsSending(true);
    try {
      await onSend?.(text);
      // Clear only on success so a failed send keeps the draft.
      setMessage("");
    } catch (error) {
      console.error("TextInput: send failed", error);
    } finally {
      setIsSending(false);
    }
  }, [message, onSend, isSending]);

  const handleKeyDown = (
    e: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void handleSend();
    }
  };

  const canSend = !disabled && !isSending && message.trim().length > 0;

  const inputProps = {
    placeholder,
    "aria-label": ariaLabel,
    value: message,
    onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
      setMessage(e.target.value),
    onKeyDown: handleKeyDown,
    disabled: disabled || isSending,
  };

  return (
    <InputGroup data-slot="text-input" className={className}>
      {multiline ? (
        <InputGroupTextarea
          ref={inputRef as React.Ref<HTMLTextAreaElement>}
          {...inputProps}
        />
      ) : (
        <InputGroupInput
          ref={inputRef as React.Ref<HTMLInputElement>}
          {...inputProps}
        />
      )}
      <InputGroupAddon
        align={multiline ? "block-end" : "inline-end"}
        // The primitive's block-end addon is justify-start; flip it for the
        // (default) right-hand send button.
        className={
          multiline && buttonPosition === "right" ? "justify-end" : undefined
        }
      >
        <InputGroupButton
          onClick={() => void handleSend()}
          disabled={!canSend}
          aria-label="Send message"
          aria-busy={isSending || undefined}
          size={buttonContent ? "xs" : "icon-xs"}
          {...buttonProps}
        >
          {isSending ? (
            <Loader2Icon className="animate-spin" />
          ) : (
            (buttonContent ?? <SendIcon />)
          )}
        </InputGroupButton>
      </InputGroupAddon>
    </InputGroup>
  );
}

const CONNECTED_STATES: TransportState[] = ["connected", "ready"];

export interface TextInputProps extends Omit<
  TextInputViewProps,
  "onSend" | "disabled"
> {
  /** Options forwarded to client.sendText(). */
  sendTextOptions?: SendTextOptions;
  /** Placeholder shown while disconnected. */
  disconnectedPlaceholder?: string;
  /** Skip injecting the sent message into the local conversation store. */
  noInject?: boolean;
  /** Called after a message is successfully sent. */
  onSent?: (message: string) => void;
  /** Disable the composer even while connected. */
  disabled?: boolean;
}

/** Requires PipecatClientProvider. Adds optimistic conversation text and sends it through the client. */
export function TextInput({
  sendTextOptions,
  disconnectedPlaceholder = "Connect to send",
  noInject = false,
  onSent,
  disabled,
  placeholder,
  ...props
}: TextInputProps) {
  const client = usePipecatClient();
  const transportState = usePipecatClientTransportState();
  const isConnected = CONNECTED_STATES.includes(transportState);
  const { injectMessage } = useConversationContext();

  const handleSend = useCallback(
    async (message: string) => {
      if (!isConnected || !client) return;

      if (!noInject) {
        injectMessage({
          role: "user",
          parts: [
            { text: message, final: true, createdAt: new Date().toISOString() },
          ],
        });
      }

      await client.sendText(message, sendTextOptions);
      onSent?.(message);
    },
    [isConnected, client, injectMessage, noInject, onSent, sendTextOptions],
  );

  return (
    <TextInputView
      onSend={handleSend}
      disabled={disabled || !isConnected}
      placeholder={isConnected ? placeholder : disconnectedPlaceholder}
      {...props}
    />
  );
}
