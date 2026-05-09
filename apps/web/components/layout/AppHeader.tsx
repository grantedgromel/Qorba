"use client";

import { Search, Download } from "lucide-react";

interface AppHeaderProps {
  fundName?: string | null;
}

const PERIODS = ["3M", "6M", "YTD", "1Y", "3Y", "5Y", "ALL"] as const;
type Period = (typeof PERIODS)[number];

interface PeriodSegmentedProps {
  value: Period;
  onChange: (p: Period) => void;
  disabled?: boolean;
}

function PeriodSegmented({ value, onChange, disabled }: PeriodSegmentedProps) {
  return (
    <div className="seg" aria-disabled={disabled}>
      {PERIODS.map((p) => (
        <button
          key={p}
          className={`seg-btn ${p === value ? "active" : ""}`}
          onClick={() => onChange(p)}
          disabled={disabled}
          type="button"
        >
          {p}
        </button>
      ))}
    </div>
  );
}

export function AppHeader({ fundName }: AppHeaderProps) {
  return (
    <header
      className="flex h-[52px] flex-shrink-0 items-center gap-3 px-4"
      style={{
        background: "var(--bg-0)",
        borderBottom: "1px solid var(--line)",
      }}
    >
      <div className="flex items-center gap-2.5">
        <span className="text-sm font-semibold" style={{ letterSpacing: "-0.01em" }}>
          Qorba
        </span>
        <span className="mono text-[11px]" style={{ color: "var(--ink-2)" }}>
          /
        </span>
        <span
          className="inline-flex h-7 items-center gap-2 rounded-md px-2 text-xs"
          style={{ color: "var(--ink-0)" }}
        >
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ background: "var(--accent)" }}
          />
          {fundName ?? "No fund selected"}
        </span>
      </div>

      <div className="flex-1" />

      <div
        className="hidden items-center gap-2 rounded-lg border px-2.5 md:flex"
        style={{
          height: 30,
          width: 280,
          borderColor: "var(--line)",
          background: "var(--bg-1)",
          color: "var(--ink-2)",
        }}
      >
        <Search size={14} strokeWidth={1.5} />
        <input
          placeholder="Search metrics, peers, factors…"
          className="flex-1 bg-transparent text-xs outline-none"
          style={{ color: "var(--ink-0)" }}
        />
        <span
          className="mono rounded-sm border px-1 text-[10px]"
          style={{ color: "var(--ink-2)", borderColor: "var(--line)" }}
        >
          ⌘K
        </span>
      </div>

      <PeriodSegmented value="ALL" onChange={() => {}} disabled />

      <button
        className="inline-flex h-[30px] items-center gap-2 rounded-md px-3 text-xs font-medium"
        style={{ background: "var(--accent)", color: "var(--accent-ink)" }}
        type="button"
      >
        <Download size={14} strokeWidth={1.5} />
        Export
      </button>
    </header>
  );
}
