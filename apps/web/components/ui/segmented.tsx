"use client";

import { cn } from "@/lib/utils";

interface SegmentedProps<T extends string> {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string }[];
  className?: string;
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  className,
}: SegmentedProps<T>) {
  return (
    <div role="radiogroup" className={cn("seg", className)}>
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          role="radio"
          aria-checked={opt.value === value}
          onClick={() => onChange(opt.value)}
          className={cn("seg-btn", opt.value === value && "active")}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
