import React from "react";
import { cva, VariantProps } from "class-variance-authority";

const alertVariants = cva("alert", {
  variants: {
    intent: {
      info: "alert-info",
      danger: "alert-danger",
    },
  },
  defaultVariants: {
    intent: "info",
  },
});

export interface AlertProps
  extends React.HTMLAttributes<HTMLElement>,
    VariantProps<typeof alertVariants> {}

export const Alert: React.FC<AlertProps> = ({ children, intent, title }) => {
  return (
    <div className={alertVariants({ intent })}>
      <span>{title}</span>
      {children}
    </div>
  );
};
