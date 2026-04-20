import * as React from "react";

import { cn } from "@/lib/utils";

export function UICard({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-border/70 bg-card/85 p-4 shadow-glow backdrop-blur-sm",
        className,
      )}
      {...props}
    />
  );
}
