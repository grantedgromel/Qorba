"use client";

import { smoothPath } from "./smoothPath";

interface MiniSparkProps {
  values: number[];
  isUp?: boolean;
  width?: number;
  height?: number;
}

export function MiniSpark({ values, isUp = true, width = 64, height = 20 }: MiniSparkProps) {
  if (!values || values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const x = (i: number) => (i / (values.length - 1)) * width;
  const y = (v: number) => 4 + (1 - (v - min) / range) * (height - 8);
  const pts = values.map<[number, number]>((v, i) => [x(i), y(v)]);
  const d = smoothPath(pts);
  const color = isUp ? "var(--pos)" : "var(--neg)";
  return (
    <svg width={width} height={height} aria-hidden className="block">
      <path d={d} fill="none" stroke={color} strokeWidth={1.4} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={x(values.length - 1)} cy={y(values[values.length - 1]!)} r={2} fill={color} />
    </svg>
  );
}
