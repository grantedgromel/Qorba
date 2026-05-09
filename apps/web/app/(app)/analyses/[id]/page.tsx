import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { AnalysisOverview } from "./AnalysisOverview";

interface MetricValue {
  metric_id: string;
  value: number | null;
  formatted: string;
}

interface TimeSeriesPoint {
  period: string;
  value: number;
}

export interface AnalysisResult {
  analysis_id: string;
  metrics: Record<string, MetricValue>;
  monthly_returns: TimeSeriesPoint[];
  cumulative_growth: TimeSeriesPoint[];
  fund_name: string;
  inception: string;
  last_observation: string;
  computed_at: string;
  version_hash: string;
}

async function compute(id: string, cookieHeader: string): Promise<AnalysisResult> {
  const apiUrl = process.env.API_URL ?? "http://localhost:8000";
  const r = await fetch(`${apiUrl}/api/v1/analyses/${id}/compute`, {
    method: "POST",
    cache: "no-store",
    headers: { cookie: cookieHeader, "content-type": "application/json" },
    body: "{}",
  });
  if (r.status === 401) redirect("/login");
  if (!r.ok) throw new Error(`Compute failed: ${r.status}`);
  return r.json();
}

export default async function AnalysisPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const cookieStore = await cookies();
  const cookieHeader = cookieStore
    .getAll()
    .map((c) => `${c.name}=${c.value}`)
    .join("; ");

  const result = await compute(id, cookieHeader);
  return <AnalysisOverview result={result} />;
}
