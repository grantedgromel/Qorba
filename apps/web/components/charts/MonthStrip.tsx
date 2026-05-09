"use client";

interface MonthStripProps {
  values: number[]; // last N decimal returns
  dates: string[]; // ISO strings, length matches values
  count?: number;
}

const fmtMonth = (iso: string): string => {
  const d = new Date(iso);
  return d.toLocaleString("en-US", { month: "short" }).toUpperCase();
};

const fmtCell = (iso: string, v: number): string =>
  `${new Date(iso).toLocaleString("en-US", { month: "short", year: "numeric" })} · ${(v * 100).toFixed(2)}%`;

export function MonthStrip({ values, dates, count = 12 }: MonthStripProps) {
  const last = values.slice(-count);
  const lastDates = dates.slice(-count);
  if (!last.length) return null;
  const maxAbs = Math.max(...last.map(Math.abs)) || 1;

  return (
    <div className="mt-4 grid grid-cols-12 gap-1">
      {last.map((v, i) => {
        const intensity = Math.min(1, Math.abs(v) / maxAbs);
        const isUp = v >= 0;
        const tint = isUp ? "var(--pos)" : "var(--neg)";
        const bg = `color-mix(in oklab, ${tint} ${(intensity * 70 + 6).toFixed(0)}%, transparent)`;
        const border = `color-mix(in oklab, ${tint} ${(intensity * 30 + 8).toFixed(0)}%, var(--line))`;
        return (
          <div
            key={lastDates[i]!}
            title={fmtCell(lastDates[i]!, v)}
            className="flex flex-col items-center gap-1 rounded-sm border px-1 py-2.5 text-center transition-transform hover:-translate-y-0.5"
            style={{ background: bg, borderColor: border }}
          >
            <div
              className="mono"
              style={{ fontSize: 9, color: "var(--ink-2)", letterSpacing: "0.06em" }}
            >
              {fmtMonth(lastDates[i]!)}
            </div>
            <div
              className="mono"
              style={{
                fontSize: 11.5,
                fontWeight: 600,
                color: tint,
                letterSpacing: "-0.02em",
              }}
            >
              {(v >= 0 ? "+" : "") + (v * 100).toFixed(1)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
