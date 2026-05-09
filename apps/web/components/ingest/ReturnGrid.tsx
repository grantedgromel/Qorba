"use client";

import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";

export interface ReturnPoint {
  period: string; // ISO date "YYYY-MM-DD"
  value: number;
}

export interface CellEdit {
  period: string;
  value: number | null;
}

interface ReturnGridProps {
  points: ReturnPoint[];
  edits: Record<string, CellEdit>;
  onEdit: (period: string, raw: string) => void;
  scale: "percent" | "decimal";
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

interface YearRow {
  year: number;
  months: (number | null)[];
}

function buildGrid(points: ReturnPoint[], edits: Record<string, CellEdit>): YearRow[] {
  const lookup = new Map<string, number | null>();
  for (const p of points) lookup.set(p.period, p.value);
  for (const e of Object.values(edits)) lookup.set(e.period, e.value);

  const all = Array.from(lookup.entries())
    .map(([period, value]) => ({ period, value }))
    .filter((p): p is { period: string; value: number | null } => p.period.length === 10);

  if (all.length === 0) return [];
  const years = new Set<number>();
  for (const p of all) years.add(Number(p.period.slice(0, 4)));
  const yearsSorted = Array.from(years).sort();
  const rows: YearRow[] = yearsSorted.map((year) => ({
    year,
    months: Array(12).fill(null),
  }));
  const rowByYear = new Map(rows.map((r) => [r.year, r]));

  for (const p of all) {
    const year = Number(p.period.slice(0, 4));
    const month = Number(p.period.slice(5, 7));
    const row = rowByYear.get(year);
    if (row && month >= 1 && month <= 12) row.months[month - 1] = p.value;
  }
  return rows;
}

function formatCell(v: number | null, scale: "percent" | "decimal"): string {
  if (v === null || !Number.isFinite(v)) return "";
  return scale === "percent" ? v.toFixed(2) : v.toFixed(4);
}

export function ReturnGrid({ points, edits, onEdit, scale }: ReturnGridProps) {
  const rows = useMemo(() => buildGrid(points, edits), [points, edits]);
  const [focused, setFocused] = useState<string | null>(null);

  if (rows.length === 0) {
    return <p className="text-sm text-muted">No data points to display.</p>;
  }

  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full font-mono text-xs tabular">
        <thead className="bg-elevated text-muted">
          <tr>
            <th className="px-2 py-2 text-left text-[10px] font-medium uppercase tracking-wider">
              Year
            </th>
            {MONTHS.map((m) => (
              <th
                key={m}
                className="px-2 py-2 text-right text-[10px] font-medium uppercase tracking-wider"
              >
                {m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.year} className="border-t border-border">
              <td className="px-2 py-1 text-fg">{row.year}</td>
              {row.months.map((cell, idx) => {
                const period = `${row.year}-${String(idx + 1).padStart(2, "0")}-01`;
                const formatted = formatCell(cell, scale);
                const isFocused = focused === period;
                return (
                  <td key={idx} className="px-1 py-0.5">
                    <input
                      value={formatted}
                      onFocus={() => setFocused(period)}
                      onBlur={() => setFocused(null)}
                      onChange={(e) => onEdit(period, e.target.value)}
                      className={cn(
                        "w-full bg-transparent px-1 py-1 text-right text-fg outline-none",
                        isFocused && "rounded-sm bg-elevated ring-1 ring-accent",
                        cell === null && !isFocused && "text-muted",
                      )}
                      placeholder="—"
                      inputMode="decimal"
                    />
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function applyEdits(
  points: ReturnPoint[],
  edits: Record<string, CellEdit>,
): { period: string; value: number | null }[] {
  const merged = new Map<string, number | null>();
  for (const p of points) merged.set(p.period, p.value);
  for (const e of Object.values(edits)) merged.set(e.period, e.value);
  return Array.from(merged.entries()).map(([period, value]) => ({ period, value }));
}
