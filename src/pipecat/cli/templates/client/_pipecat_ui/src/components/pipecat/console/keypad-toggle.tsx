"use client";

import { PhoneIcon } from "lucide-react";
import * as React from "react";

import { useMinWidth } from "@/components/pipecat/console/panel";
import {
  DTMFKeypad,
  type DTMFKeypadMode,
} from "@/components/pipecat/dtmf-keypad";
import { Button } from "@/components/ui/button";
import {
  Drawer,
  DrawerContent,
  DrawerDescription,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

export interface ConsoleKeypadToggleProps {
  /** How the keypad dispatches presses (default "buffered"). */
  mode?: DTMFKeypadMode;
  /** Overrides merged onto the keypad (wins over defaults). */
  keypadProps?: Partial<React.ComponentProps<typeof DTMFKeypad>>;
  className?: string;
}

/**
 * Header button opening the DTMF keypad — a popover on desktop, a drawer on
 * mobile. One instantiation: the surface switches on the viewport instead
 * of rendering both. Must be rendered inside a PipecatClientProvider.
 */
export function ConsoleKeypadToggle({
  mode = "buffered",
  keypadProps,
  className,
}: ConsoleKeypadToggleProps) {
  const isDesktop = useMinWidth(640);

  const trigger = (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label="Open keypad"
      className={className}
      data-slot="console-keypad-toggle"
    >
      <PhoneIcon />
    </Button>
  );

  if (isDesktop) {
    return (
      <Popover>
        <PopoverTrigger render={trigger} />
        <PopoverContent align="end" side="bottom" className="w-auto">
          <DTMFKeypad mode={mode} {...keypadProps} />
        </PopoverContent>
      </Popover>
    );
  }

  return (
    <Drawer>
      <DrawerTrigger render={trigger} />
      <DrawerContent>
        <DrawerHeader>
          <DrawerTitle>Keypad</DrawerTitle>
          <DrawerDescription className="sr-only">
            Send DTMF tones to the connected bot.
          </DrawerDescription>
        </DrawerHeader>
        <div className="mx-auto w-full max-w-xs p-4 pt-0">
          <DTMFKeypad mode={mode} {...keypadProps} />
        </div>
      </DrawerContent>
    </Drawer>
  );
}
