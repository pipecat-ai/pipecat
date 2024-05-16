import React from "react";
import { cva, VariantProps } from "class-variance-authority";

const buttonVariants = cva("button", {
  variants: {
    variant: {
      primary: "button-primary",
      ghost: "button-ghost",
    },
    size: {
      base: "",
      icon: "button-icon",
    },
  },
  defaultVariants: {
    variant: "primary",
    size: "base",
  },
});

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button: React.FC<ButtonProps> = ({ variant, size, ...props }) => {
  return <button className={buttonVariants({ variant, size })} {...props} />;
};
