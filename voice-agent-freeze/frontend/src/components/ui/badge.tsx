import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-slate-700 text-slate-100",
        secondary: "border-slate-600 bg-slate-800 text-slate-200",
        user: "border-sky-400/40 bg-sky-500/20 text-sky-300",
        assistant: "border-emerald-400/40 bg-emerald-500/20 text-emerald-300",
        interrupted: "border-rose-400/40 bg-rose-500/20 text-rose-300",
        latency: "border-rose-400/40 bg-rose-500/20 text-rose-300",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
