"use client";

import { useEffect, useRef, useState } from "react";
import { smoothPath } from "./smoothPath";

interface HeroChartProps {
  values: number[];
  dates: string[]; // ISO strings
  height?: number;
  isUp: boolean;
}

const fmtMonth = (iso: string): string => {
  const d = new Date(iso);
  return d
    .toLocaleString("en-US", { month: "short", year: "2-digit" })
    .toUpperCase();
};

export function HeroChart({ values, dates, height = 340, isUp }: HeroChartProps) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [w, setW] = useState(900);
  const [hover, setHover] = useState<number | null>(null);
  const [drawn, setDrawn] = useState(false);

  useEffect(() => {
    if (!wrapRef.current) return;
    const ro = new ResizeObserver(([e]) => setW(e!.contentRect.width));
    ro.observe(wrapRef.current);
    setW(wrapRef.current.clientWidth);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    setDrawn(false);
    const t = setTimeout(() => setDrawn(true), 30);
    return () => clearTimeout(t);
  }, [values.length, isUp]);

  if (values.length < 2) {
    return (
      <div ref={wrapRef} style={{ width: "100%", height }} className="grid place-items-center">
        <span className="eyebrow">Not enough data</span>
      </div>
    );
  }

  const padL = 0;
  const padR = 0;
  const padT = 22;
  const padB = 28;
  const W = Math.max(320, w);
  const H = height;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const minY = Math.min(...values);
  const maxY = Math.max(...values);
  const range = maxY - minY || 1;
  const yMin = minY - range * 0.06;
  const yMax = maxY + range * 0.1;
  const x = (i: number) => padL + (i / (values.length - 1)) * innerW;
  const y = (v: number) => padT + (1 - (v - yMin) / (yMax - yMin)) * innerH;

  const lineColor = isUp ? "var(--pos)" : "var(--neg)";
  const pts = values.map<[number, number]>((v, i) => [x(i), y(v)]);
  const linePath = smoothPath(pts);
  const areaPath = `${linePath} L ${x(values.length - 1).toFixed(2)},${y(yMin).toFixed(2)} L ${x(0).toFixed(2)},${y(yMin).toFixed(2)} Z`;

  const refTicks = [yMin + range * 0.1, yMin + range * 0.4, yMin + range * 0.7, maxY];
  const xTicks = [0, Math.floor(values.length * 0.5), values.length - 1];

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    const px = e.clientX - r.left;
    const i = Math.max(
      0,
      Math.min(values.length - 1, Math.round(((px - padL) / innerW) * (values.length - 1))),
    );
    setHover(i);
  };

  const lastX = x(values.length - 1);
  const lastY = y(values[values.length - 1]!);

  return (
    <div ref={wrapRef} style={{ width: "100%", position: "relative" }}>
      <svg
        width={W}
        height={H}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        style={{ display: "block", overflow: "visible" }}
      >
        <defs>
          <linearGradient id="hero-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={lineColor} stopOpacity="0.18" />
            <stop offset="60%" stopColor={lineColor} stopOpacity="0.04" />
            <stop offset="100%" stopColor={lineColor} stopOpacity="0" />
          </linearGradient>
          <clipPath id="hero-clip">
            <rect
              x="0"
              y="0"
              width={drawn ? innerW : 0}
              height={H}
              style={{
                transition: "width 1100ms cubic-bezier(0.22, 1, 0.36, 1)",
              }}
            />
          </clipPath>
        </defs>

        {refTicks.map((t, i) => (
          <line
            key={i}
            x1={padL}
            x2={W - padR}
            y1={y(t)}
            y2={y(t)}
            stroke="var(--line-soft)"
            strokeWidth={1}
          />
        ))}

        <g clipPath="url(#hero-clip)">
          <path d={areaPath} fill="url(#hero-fill)" />
          <path
            d={linePath}
            fill="none"
            stroke={lineColor}
            strokeWidth={1.75}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </g>

        {drawn && (
          <g style={{ transition: "opacity 400ms ease 1100ms" }}>
            <circle cx={lastX} cy={lastY} r={14} fill={lineColor} opacity={0.1}>
              <animate attributeName="r" values="6;16;6" dur="2.4s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.20;0;0.20" dur="2.4s" repeatCount="indefinite" />
            </circle>
            <circle cx={lastX} cy={lastY} r={3.5} fill={lineColor} />
          </g>
        )}

        {xTicks.map((i, k) => (
          <text
            key={k}
            x={k === 0 ? padL + 2 : k === xTicks.length - 1 ? W - padR - 2 : x(i)}
            y={H - 6}
            textAnchor={k === 0 ? "start" : k === xTicks.length - 1 ? "end" : "middle"}
            fontSize={10.5}
            fontFamily="var(--font-mono)"
            fill="var(--ink-2)"
            letterSpacing="0.06em"
          >
            {fmtMonth(dates[i]!)}
          </text>
        ))}

        {hover != null && (
          <g>
            <line
              x1={x(hover)}
              x2={x(hover)}
              y1={padT}
              y2={H - padB}
              stroke="var(--ink-2)"
              strokeWidth={1}
              strokeDasharray="2 4"
              opacity={0.45}
            />
            <circle
              cx={x(hover)}
              cy={y(values[hover]!)}
              r={4}
              fill="var(--bg-0)"
              stroke={lineColor}
              strokeWidth={1.75}
            />
          </g>
        )}
      </svg>

      {hover != null && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: Math.max(0, Math.min(x(hover) - 60, W - 140)),
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--ink-0)",
            background: "var(--bg-1)",
            border: "1px solid var(--line)",
            borderRadius: 6,
            padding: "6px 10px",
            pointerEvents: "none",
            whiteSpace: "nowrap",
            boxShadow: "0 4px 18px rgba(0,0,0,0.18)",
            minWidth: 130,
          }}
        >
          <div className="eyebrow" style={{ marginBottom: 3 }}>
            {new Date(dates[hover]!).toLocaleString("en-US", { month: "short", year: "numeric" })}
          </div>
          <div style={{ fontSize: 14, fontWeight: 500 }}>${values[hover]!.toFixed(2)}</div>
          <div style={{ fontSize: 10, color: "var(--ink-2)", marginTop: 1 }}>
            <span style={{ color: values[hover]! >= values[0]! ? "var(--pos)" : "var(--neg)" }}>
              {(values[hover]! >= values[0]! ? "+" : "") +
                (((values[hover]! / values[0]!) - 1) * 100).toFixed(2)}
              %
            </span>{" "}
            from start
          </div>
        </div>
      )}
    </div>
  );
}
