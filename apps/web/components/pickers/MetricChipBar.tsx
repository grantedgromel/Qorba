"use client";

import { X } from "lucide-react";
import type { MetricCatalogEntry } from "@/app/(app)/analyses/[id]/page";

interface MetricChipBarProps {
  catalog: MetricCatalogEntry[];
  selected: string[];
  onRemove: (id: string) => void;
  trailing?: React.ReactNode;
}

export function MetricChipBar({
  catalog,
  selected,
  onRemove,
  trailing,
}: MetricChipBarProps) {
  const byId = new Map(catalog.map((c) => [c.id, c]));
  if (selected.length === 0) {
    return (
      <div className="flex items-center gap-2 text-xs" style={{ color: "var(--ink-2)" }}>
        No metrics selected.
        {trailing}
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {selected.map((id) => {
        const item = byId.get(id);
        const label = item?.label ?? id;
        return (
          <button
            key={id}
            type="button"
            onClick={() => onRemove(id)}
            className="chip group inline-flex items-center gap-1.5 transition-colors"
            style={{ borderColor: "var(--line)", color: "var(--ink-1)" }}
            title={`Remove ${label}`}
          >
            <span>{label}</span>
            <X
              size={11}
              strokeWidth={1.75}
              className="opacity-50 transition-opacity group-hover:opacity-100"
            />
          </button>
        );
      })}
      {trailing}
    </div>
  );
}
