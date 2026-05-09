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

export type Period = "3M" | "6M" | "YTD" | "1Y" | "3Y" | "5Y" | "ALL";

export interface AnalysisResult {
  analysis_id: string;
  metrics: Record<string, MetricValue>;
  monthly_returns: TimeSeriesPoint[];
  cumulative_growth: TimeSeriesPoint[];
  fund_name: string;
  inception: string;
  last_observation: string;
  period: Period;
  computed_at: string;
  version_hash: string;
}

export interface MetricCatalogEntry {
  id: string;
  label: string;
  group: string;
  default: boolean;
  requires_benchmark: boolean;
}

async function fetchJson<T>(url: string, cookieHeader: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, {
    cache: "no-store",
    ...init,
    headers: { cookie: cookieHeader, ...(init?.headers ?? {}) },
  });
  if (r.status === 401) redirect("/login");
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json() as Promise<T>;
}

const VALID_PERIODS: Period[] = ["3M", "6M", "YTD", "1Y", "3Y", "5Y", "ALL"];

function parsePeriod(raw: string | undefined): Period {
  return VALID_PERIODS.includes(raw as Period) ? (raw as Period) : "ALL";
}

export default async function AnalysisPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ period?: string; metrics?: string }>;
}) {
  const { id } = await params;
  const sp = await searchParams;
  const period = parsePeriod(sp.period);
  const metricsParam = sp.metrics ? `&metric_ids=${encodeURIComponent(sp.metrics)}` : "";

  const cookieStore = await cookies();
  const cookieHeader = cookieStore
    .getAll()
    .map((c) => `${c.name}=${c.value}`)
    .join("; ");

  const apiUrl = process.env.API_URL ?? "http://localhost:8000";

  const [result, catalog] = await Promise.all([
    fetchJson<AnalysisResult>(
      `${apiUrl}/api/v1/analyses/${id}/compute?period=${period}${metricsParam}`,
      cookieHeader,
      { method: "POST", body: "{}", headers: { "content-type": "application/json" } },
    ),
    fetchJson<{ items: MetricCatalogEntry[] }>(
      `${apiUrl}/api/v1/metrics/catalog`,
      cookieHeader,
    ),
  ]);

  return (
    <AnalysisOverview
      analysisId={id}
      initial={result}
      catalog={catalog.items}
    />
  );
}
