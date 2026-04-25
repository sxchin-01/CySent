import * as React from "react";

import { cn } from "@/lib/utils";

export function UICard({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("dt-panel p-4", className)} {...props} />
  );
}
