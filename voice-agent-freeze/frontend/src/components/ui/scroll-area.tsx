import * as React from "react";
import { cn } from "@/lib/utils";

function ScrollArea({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("relative overflow-hidden", className)} {...props}>
      <div className="h-full overflow-y-auto pr-1">{children}</div>
    </div>
  );
}

export { ScrollArea };
