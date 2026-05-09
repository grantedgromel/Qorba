"use client";

import { useMemo } from "react";

interface SparklineProps {
  values: number[];
  width?: number;
  height?: number;
  className?: string;
}

/**
 * Cumulative-growth sparkline. Compounds the input series and draws a
 * filled area under the curve. Pure SVG, no chart lib — until Sprint 3
 * brings in Visx, this is the placeholder.
 */
export function Sparkline({
  values,
  width = 600,
  height = 80,
  className,
}: SparklineProps) {
  const path = useMemo(() => {
    if (!values.length) return null;
    let cum = 1;
    const ys: number[] = [];
    for (const v of values) {
      cum *= 1 + v;
      ys.push(cum);
    }
    const min = Math.min(...ys, 1);
    const max = Math.max(...ys, 1);
    const range = max - min || 1;
    const stepX = width / Math.max(values.length - 1, 1);
    const points = ys.map(
      (y, i) => `${i * stepX},${height - ((y - min) / range) * height}`,
    );
    const line = `M${points.join(" L")}`;
    const area = `${line} L${(values.length - 1) * stepX},${height} L0,${height} Z`;
    return { line, area };
  }, [values, width, height]);

  if (!path) {
    return (
      <div
        className={className}
        style={{ width, height }}
        aria-label="Sparkline (empty)"
      />
    );
  }

  return (
    <svg
      role="img"
      aria-label="Cumulative growth sparkline"
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      height={height}
      className={className}
      preserveAspectRatio="none"
    >
      <path d={path.area} fill="hsl(var(--accent) / 0.15)" />
      <path d={path.line} fill="none" stroke="hsl(var(--accent))" strokeWidth={1.5} />
    </svg>
  );
}
