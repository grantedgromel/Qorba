/**
 * Catmull-Rom → cubic Bezier path generator. Lifted from the Qorba design
 * handoff. Produces a single SVG `d` attribute string for an array of
 * [x, y] points.
 */
export function smoothPath(pts: [number, number][], tension = 1): string {
  if (!pts || pts.length < 2) return "";
  if (pts.length === 2) {
    const a = pts[0]!;
    const b = pts[1]!;
    return `M ${a[0]},${a[1]} L ${b[0]},${b[1]}`;
  }
  const at = (i: number) => pts[Math.max(0, Math.min(pts.length - 1, i))]!;
  let d = `M ${pts[0]![0].toFixed(2)},${pts[0]![1].toFixed(2)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = at(i - 1);
    const p1 = at(i);
    const p2 = at(i + 1);
    const p3 = at(i + 2);
    const cp1x = p1[0] + ((p2[0] - p0[0]) / 6) * tension;
    const cp1y = p1[1] + ((p2[1] - p0[1]) / 6) * tension;
    const cp2x = p2[0] - ((p3[0] - p1[0]) / 6) * tension;
    const cp2y = p2[1] - ((p3[1] - p1[1]) / 6) * tension;
    d += ` C ${cp1x.toFixed(2)},${cp1y.toFixed(2)} ${cp2x.toFixed(2)},${cp2y.toFixed(2)} ${p2[0].toFixed(2)},${p2[1].toFixed(2)}`;
  }
  return d;
}
