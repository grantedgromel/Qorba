import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface MetricValue {
  metric_id: string;
  value: number | null;
  formatted: string;
}

interface AnalysisResult {
  analysis_id: string;
  metrics: Record<string, MetricValue>;
  computed_at: string;
  version_hash: string;
}

const METRIC_LABELS: Record<string, string> = {
  ann_return_geo: "Annualized Return",
  ann_vol: "Annualized Volatility",
  sharpe: "Sharpe Ratio",
  sortino: "Sortino Ratio",
  max_dd: "Max Drawdown",
  win_rate: "Win Rate",
};

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

export default async function AnalysisPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const cookieStore = await cookies();
  const cookieHeader = cookieStore
    .getAll()
    .map((c) => `${c.name}=${c.value}`)
    .join("; ");

  const result = await compute(id, cookieHeader);
  const metrics = Object.values(result.metrics);

  return (
    <div className="container max-w-5xl space-y-8 py-12">
      <div className="space-y-1">
        <p className="text-xs uppercase tracking-wider text-muted">Analysis</p>
        <h1 className="text-2xl font-semibold tracking-tight">Performance overview</h1>
        <p className="text-xs text-muted">Computed {new Date(result.computed_at).toLocaleString()}</p>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
        {metrics.map((m) => (
          <Card key={m.metric_id}>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted">
                {METRIC_LABELS[m.metric_id] ?? m.metric_id}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="font-mono text-2xl tabular tracking-tight">{m.formatted}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
