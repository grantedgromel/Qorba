"use client";

import { useMemo } from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { AnimatedNumber } from "@/components/charts/AnimatedNumber";
import { HeroChart } from "@/components/charts/HeroChart";
import { MiniSpark } from "@/components/charts/MiniSpark";
import { MonthStrip } from "@/components/charts/MonthStrip";
import type { AnalysisResult } from "./page";

const PERIODS = ["3M", "6M", "YTD", "1Y", "3Y", "5Y", "ALL"] as const;
type Period = (typeof PERIODS)[number];

const fmtMonthYear = (iso: string): string =>
  new Date(iso).toLocaleString("en-US", { month: "short", year: "numeric" });

interface QuietStatProps {
  label: string;
  value: string;
  sub?: string;
  spark?: number[];
  isUp?: boolean;
  accent?: "pos" | "neg" | undefined;
  last?: boolean;
}

function QuietStat({ label, value, sub, spark, isUp, accent, last }: QuietStatProps) {
  const color =
    accent === "pos" ? "var(--pos)" : accent === "neg" ? "var(--neg)" : "var(--ink-0)";
  return (
    <div
      className="flex flex-col gap-2.5 px-7 py-5"
      style={{ borderRight: last ? "none" : "1px solid var(--line-soft)" }}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="eyebrow">{label}</span>
        {spark && spark.length > 1 && <MiniSpark values={spark} isUp={!!isUp} width={64} height={20} />}
      </div>
      <div
        className="mono"
        style={{
          fontSize: 26,
          fontWeight: 500,
          letterSpacing: "-0.025em",
          color,
          lineHeight: 1,
        }}
      >
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 11.5, color: "var(--ink-2)", lineHeight: 1.4 }}>{sub}</div>
      )}
    </div>
  );
}

interface MicroStatProps {
  label: string;
  value: string;
  accent?: "pos" | "neg";
}

function MicroStat({ label, value, accent }: MicroStatProps) {
  const color = accent === "pos" ? "var(--pos)" : accent === "neg" ? "var(--neg)" : "var(--ink-0)";
  return (
    <div className="flex flex-col gap-1">
      <span className="eyebrow" style={{ fontSize: 9.5 }}>
        {label}
      </span>
      <span
        className="mono"
        style={{ fontSize: 17, fontWeight: 500, color, letterSpacing: "-0.02em" }}
      >
        {value}
      </span>
    </div>
  );
}

interface PeriodRailProps {
  period: Period;
  onPick: (p: Period) => void;
}

function PeriodRail({ period, onPick }: PeriodRailProps) {
  return (
    <div className="inline-flex" style={{ gap: 22 }}>
      {PERIODS.map((p) => {
        const active = p === period;
        return (
          <button
            key={p}
            onClick={() => onPick(p)}
            className="mono relative py-1.5"
            style={{
              fontSize: 11,
              letterSpacing: "0.1em",
              color: active ? "var(--ink-0)" : "var(--ink-2)",
              fontWeight: active ? 600 : 500,
              transition: "color 200ms",
              background: "transparent",
              border: 0,
              cursor: "pointer",
              padding: "6px 0",
            }}
          >
            {p}
            <span
              aria-hidden
              style={{
                position: "absolute",
                left: 0,
                right: 0,
                bottom: 0,
                height: 1,
                background: "var(--ink-0)",
                transformOrigin: "center",
                transform: active ? "scaleX(1)" : "scaleX(0)",
                transition: "transform 260ms cubic-bezier(0.22,1,0.36,1)",
              }}
            />
          </button>
        );
      })}
    </div>
  );
}

interface EditorialItem {
  kind: "pos" | "neg" | "warn" | "info";
  kicker: string;
  title: string;
  body: string;
}

function Editorial({ items }: { items: EditorialItem[] }) {
  const tint = (k: EditorialItem["kind"]): string =>
    k === "pos"
      ? "var(--pos)"
      : k === "neg"
        ? "var(--neg)"
        : k === "warn"
          ? "var(--warn)"
          : "var(--info)";
  return (
    <div className="flex flex-col" style={{ gap: 26 }}>
      {items.map((it, i) => (
        <div key={i} className="grid items-start gap-3.5" style={{ gridTemplateColumns: "14px 1fr" }}>
          <span
            aria-hidden
            className="inline-block h-1.5 w-1.5 rounded-full"
            style={{ background: tint(it.kind), marginTop: 8 }}
          />
          <div className="flex flex-col gap-1.5">
            <span
              className="mono"
              style={{
                fontSize: 10,
                letterSpacing: "0.12em",
                color: tint(it.kind),
                textTransform: "uppercase",
                fontWeight: 500,
              }}
            >
              {it.kicker}
            </span>
            <span
              style={{
                fontSize: 14.5,
                fontWeight: 500,
                color: "var(--ink-0)",
                letterSpacing: "-0.012em",
                lineHeight: 1.35,
              }}
            >
              {it.title}
            </span>
            <span style={{ fontSize: 12.5, color: "var(--ink-1)", lineHeight: 1.55 }}>
              {it.body}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

export function AnalysisOverview({ result }: { result: AnalysisResult }) {
  const growth = result.cumulative_growth;
  const monthly = result.monthly_returns;

  const dollars = growth.length ? growth[growth.length - 1]!.value : 100;
  const startDollars = growth.length ? growth[0]!.value : 100;
  const periodReturn = startDollars ? dollars / startDollars - 1 : 0;
  const isUp = periodReturn >= 0;

  const dts = useMemo(() => {
    // Prepend a synthetic prior point so the chart starts at $100 anchor.
    if (!growth.length) return [];
    const first = new Date(growth[0]!.period);
    const prior = new Date(first.getFullYear(), first.getMonth() - 1, 1).toISOString().slice(0, 10);
    return [prior, ...growth.map((g) => g.period)];
  }, [growth]);

  const values = useMemo(() => (growth.length ? [100, ...growth.map((g) => g.value)] : []), [growth]);

  const sharpe = result.metrics.sharpe?.value ?? null;
  const annVol = result.metrics.ann_vol?.value ?? null;
  const maxDD = result.metrics.max_dd?.value ?? null;
  const winRate = result.metrics.win_rate?.value ?? null;

  const positives = monthly.filter((m) => m.value > 0);
  const negatives = monthly.filter((m) => m.value < 0);
  const bestMonth = monthly.length ? Math.max(...monthly.map((m) => m.value)) : 0;
  const worstMonth = monthly.length ? Math.min(...monthly.map((m) => m.value)) : 0;
  const avgGain = positives.length
    ? positives.reduce((s, m) => s + m.value, 0) / positives.length
    : 0;
  const avgLoss = negatives.length
    ? negatives.reduce((s, m) => s + m.value, 0) / negatives.length
    : 0;

  const sharpeSpark = useMemo(() => {
    if (monthly.length < 7) return [];
    const win = 6;
    const out: number[] = [];
    for (let i = win; i <= monthly.length; i++) {
      const slice = monthly.slice(i - win, i).map((m) => m.value);
      const mean = slice.reduce((s, v) => s + v, 0) / win;
      const sd = Math.sqrt(
        slice.reduce((s, v) => s + (v - mean) * (v - mean), 0) / (win - 1),
      );
      out.push(sd ? (mean * 12) / (sd * Math.sqrt(12)) : 0);
    }
    return out;
  }, [monthly]);

  const volSpark = useMemo(() => {
    if (monthly.length < 7) return [];
    const win = 6;
    const out: number[] = [];
    for (let i = win; i <= monthly.length; i++) {
      const slice = monthly.slice(i - win, i).map((m) => m.value);
      const mean = slice.reduce((s, v) => s + v, 0) / win;
      const sd = Math.sqrt(
        slice.reduce((s, v) => s + (v - mean) * (v - mean), 0) / (win - 1),
      );
      out.push(sd * Math.sqrt(12));
    }
    return out;
  }, [monthly]);

  const fmtPct = (v: number, digits = 2, signed = false): string => {
    const s = (v * 100).toFixed(digits);
    return (signed && v > 0 ? "+" : "") + s + "%";
  };

  return (
    <div
      className="page-enter mx-auto flex max-w-[1100px] flex-col px-7"
      style={{ gap: 56, paddingTop: 32, paddingBottom: 24 }}
    >
      {/* HERO */}
      <section className="flex flex-col" style={{ gap: 28 }}>
        <div className="flex flex-wrap items-end justify-between gap-8">
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2.5">
              <span
                className="inline-block h-1.5 w-1.5 rounded-full"
                style={{
                  background: "var(--pos)",
                  boxShadow:
                    "0 0 0 4px color-mix(in oklab, var(--pos) 18%, transparent)",
                }}
              />
              <span
                className="eyebrow"
                style={{ fontSize: 10.5, letterSpacing: "0.16em" }}
              >
                {result.fund_name.toUpperCase()} · LIVE
              </span>
              <span
                className="mono"
                style={{ fontSize: 10.5, letterSpacing: "0.06em", color: "var(--ink-2)" }}
              >
                · COMPUTED {new Date(result.computed_at).toLocaleDateString()}
              </span>
            </div>

            <div
              className="mono"
              style={{
                fontSize: 88,
                fontWeight: 300,
                letterSpacing: "-0.04em",
                lineHeight: 0.95,
                color: "var(--ink-0)",
              }}
            >
              $<AnimatedNumber value={dollars} format={(v) => v.toFixed(2)} />
            </div>

            <div className="flex flex-wrap items-center gap-3.5">
              <span
                className="mono"
                style={{
                  fontSize: 18,
                  fontWeight: 500,
                  letterSpacing: "-0.02em",
                  color: isUp ? "var(--pos)" : "var(--neg)",
                }}
              >
                <AnimatedNumber
                  value={periodReturn}
                  format={(v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%"}
                />
              </span>
              <span style={{ width: 1, height: 14, background: "var(--line)" }} />
              <span style={{ fontSize: 12, color: "var(--ink-2)" }}>
                Growth of{" "}
                <span className="mono" style={{ color: "var(--ink-0)" }}>
                  $100
                </span>{" "}
                · {fmtMonthYear(result.inception)} → {fmtMonthYear(result.last_observation)}
              </span>
            </div>
          </div>

          <PeriodRail period="ALL" onPick={() => {}} />
        </div>

        <HeroChart values={values} dates={dts} height={340} isUp={isUp} />
      </section>

      {/* QUIET STATS */}
      <section
        className="grid grid-cols-2 sm:grid-cols-4"
        style={{
          margin: "0 -28px",
          borderTop: "1px solid var(--line-soft)",
          borderBottom: "1px solid var(--line-soft)",
        }}
      >
        <QuietStat
          label="SHARPE"
          value={sharpe == null ? "—" : sharpe.toFixed(2)}
          sub="Risk-adjusted return · rf 0%"
          spark={sharpeSpark}
          isUp={(sharpe ?? 0) >= 1}
        />
        <QuietStat
          label="VOLATILITY"
          value={annVol == null ? "—" : fmtPct(annVol, 1)}
          sub="Annualized · 6-mo rolling"
          spark={volSpark}
          isUp={false}
        />
        <QuietStat
          label="MAX DRAWDOWN"
          value={maxDD == null ? "—" : fmtPct(maxDD, 1)}
          sub="Peak-to-trough"
          accent="neg"
        />
        <QuietStat
          label="WIN RATE"
          value={winRate == null ? "—" : fmtPct(winRate, 0)}
          sub={`${positives.length}/${monthly.length} positive months`}
          last
        />
      </section>

      {/* MONTH STRIP + EDITORIAL */}
      <section
        className="grid items-start gap-14"
        style={{ gridTemplateColumns: "1.45fr 1fr" }}
      >
        <div>
          <div className="flex items-baseline justify-between">
            <span
              className="text-base font-medium"
              style={{ letterSpacing: "-0.012em" }}
            >
              Last 12 months
            </span>
            <Link
              href="/calendar"
              className="inline-flex items-center gap-1 text-xs"
              style={{ color: "var(--ink-2)" }}
            >
              View calendar <ArrowRight size={12} strokeWidth={1.5} />
            </Link>
          </div>
          <div className="text-xs" style={{ color: "var(--ink-2)" }}>
            Monthly returns · color intensity scales with magnitude
          </div>
          <MonthStrip
            values={monthly.map((m) => m.value)}
            dates={monthly.map((m) => m.period)}
          />

          <div
            className="mt-8 grid grid-cols-2 gap-6 pt-5 sm:grid-cols-4"
            style={{ borderTop: "1px solid var(--line-soft)" }}
          >
            <MicroStat label="BEST MONTH" value={fmtPct(bestMonth, 2, true)} accent="pos" />
            <MicroStat label="WORST MONTH" value={fmtPct(worstMonth, 2, true)} accent="neg" />
            <MicroStat label="AVG GAIN" value={fmtPct(avgGain, 2, true)} />
            <MicroStat label="AVG LOSS" value={fmtPct(avgLoss, 2, true)} />
          </div>
        </div>

        <div>
          <div className="mb-4">
            <div
              className="text-base font-medium"
              style={{ letterSpacing: "-0.012em" }}
            >
              What changed
            </div>
            <div className="mt-0.5 text-xs" style={{ color: "var(--ink-2)" }}>
              Auto-generated · Sprint 6 will source these from rolling exposures
            </div>
          </div>
          <Editorial
            items={[
              {
                kind: isUp ? "pos" : "neg",
                kicker: "RETURN",
                title: isUp ? "Period closed in the green" : "Period drawdown — review allocation",
                body: `${fmtPct(periodReturn, 2, true)} on $100 since ${fmtMonthYear(result.inception)}, with a final mark of $${dollars.toFixed(2)}.`,
              },
              {
                kind: (sharpe ?? 0) >= 1 ? "pos" : "warn",
                kicker: "SHARPE",
                title: (sharpe ?? 0) >= 1 ? "Risk-adjusted profile holds up" : "Risk-adjusted return below 1.0",
                body:
                  sharpe == null
                    ? "Insufficient data to compute Sharpe."
                    : `Sharpe of ${sharpe.toFixed(2)} on ${monthly.length} months. Annualized vol ${annVol == null ? "—" : fmtPct(annVol, 1)}.`,
              },
              {
                kind: maxDD != null && maxDD <= -0.1 ? "neg" : "info",
                kicker: "DRAWDOWN",
                title:
                  maxDD != null && maxDD <= -0.1
                    ? "Material drawdown in the period"
                    : "No material drawdown",
                body:
                  maxDD == null
                    ? "Drawdown not computed."
                    : `Worst peak-to-trough was ${fmtPct(maxDD, 1)}. Compounded recovery ran ${monthly.length} months.`,
              },
            ]}
          />
        </div>
      </section>

      {/* FOOTER */}
      <section
        className="grid grid-cols-2 gap-6 pt-5 sm:grid-cols-4"
        style={{ borderTop: "1px solid var(--line-soft)" }}
      >
        {[
          {
            label: "DATA RANGE",
            value: `${fmtMonthYear(result.inception)} – ${fmtMonthYear(result.last_observation)}`,
            sub: `${monthly.length} months`,
          },
          {
            label: "BENCHMARK",
            value: "Not attached",
            sub: "Add via /benchmarks/import",
          },
          { label: "RISK-FREE", value: "0.00%", sub: "annualized · editable" },
          {
            label: "LATEST",
            value: monthly.length
              ? fmtPct(monthly[monthly.length - 1]!.value, 2, true)
              : "—",
            sub: monthly.length ? fmtMonthYear(monthly[monthly.length - 1]!.period) : "",
          },
        ].map((f) => (
          <div key={f.label} className="flex flex-col gap-1">
            <span className="eyebrow">{f.label}</span>
            <span
              className="mono"
              style={{
                fontSize: 14,
                fontWeight: 500,
                color: "var(--ink-0)",
                letterSpacing: "-0.015em",
              }}
            >
              {f.value}
            </span>
            <span style={{ fontSize: 11, color: "var(--ink-2)" }}>{f.sub}</span>
          </div>
        ))}
      </section>
    </div>
  );
}
