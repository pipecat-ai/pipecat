import React from "react";
import { cva, VariantProps } from "class-variance-authority";
import { CircleAlert } from "lucide-react";

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
      <h3>
        {intent === "danger" && <CircleAlert size={18} />}
        {title}
      </h3>
      {children}
    </div>
  );
};
