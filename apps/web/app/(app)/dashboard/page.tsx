import { cookies } from "next/headers";
import Link from "next/link";
import { redirect } from "next/navigation";

interface AnalysisOut {
  id: string;
  fund_id: string;
  created_at: string;
  metrics: { metric_ids: string[] };
}

interface FundOut {
  id: string;
  name: string;
  inception: string;
  last_observation: string;
  n_observations: number;
}

async function fetchAnalyses(cookieHeader: string): Promise<AnalysisOut[]> {
  const apiUrl = process.env.API_URL ?? "http://localhost:8000";
  const r = await fetch(`${apiUrl}/api/v1/analyses`, {
    cache: "no-store",
    headers: { cookie: cookieHeader },
  });
  if (r.status === 401) redirect("/login");
  if (!r.ok) return [];
  return r.json();
}

async function fetchFund(id: string, cookieHeader: string): Promise<FundOut | null> {
  const apiUrl = process.env.API_URL ?? "http://localhost:8000";
  const r = await fetch(`${apiUrl}/api/v1/funds/${id}`, {
    cache: "no-store",
    headers: { cookie: cookieHeader },
  });
  return r.ok ? r.json() : null;
}

const fmtRange = (a: string, b: string): string => {
  const fmt = (s: string) =>
    new Date(s).toLocaleString("en-US", { month: "short", year: "2-digit" });
  return `${fmt(a)} → ${fmt(b)}`;
};

export default async function DashboardPage() {
  const cookieStore = await cookies();
  const cookieHeader = cookieStore
    .getAll()
    .map((c) => `${c.name}=${c.value}`)
    .join("; ");

  const analyses = await fetchAnalyses(cookieHeader);
  const funds = await Promise.all(
    analyses.map((a) => fetchFund(a.fund_id, cookieHeader)),
  );

  return (
    <div className="page-enter mx-auto max-w-5xl px-7 py-12">
      <header className="mb-10 flex items-end justify-between gap-6">
        <div>
          <p className="eyebrow">Recent</p>
          <h1
            className="mt-1 text-2xl font-medium"
            style={{ letterSpacing: "-0.018em" }}
          >
            Analyses
          </h1>
          <p className="mt-1 text-xs" style={{ color: "var(--ink-2)" }}>
            Pick a fund to dive in, or start a new one.
          </p>
        </div>
        <Link
          href="/analyses/new"
          className="inline-flex h-[30px] items-center gap-2 rounded-md px-3 text-xs font-medium transition-colors"
          style={{ background: "var(--accent)", color: "var(--accent-ink)" }}
        >
          New analysis
        </Link>
      </header>

      {analyses.length === 0 ? (
        <div
          className="card p-10 text-center"
          style={{ borderColor: "var(--line)" }}
        >
          <p className="font-serif text-2xl italic" style={{ color: "var(--ink-1)" }}>
            Nothing yet.
          </p>
          <p className="mt-2 text-sm" style={{ color: "var(--ink-2)" }}>
            Drop a tearsheet PDF, paste a return table, or upload a CSV.
          </p>
          <Link
            href="/analyses/new"
            className="mt-6 inline-flex h-[30px] items-center gap-2 rounded-md px-3 text-xs font-medium"
            style={{ background: "var(--accent)", color: "var(--accent-ink)" }}
          >
            Get started
          </Link>
        </div>
      ) : (
        <ul className="divide-y" style={{ borderColor: "var(--line-soft)" }}>
          {analyses.map((a, i) => {
            const fund = funds[i];
            return (
              <li key={a.id}>
                <Link
                  href={`/analyses/${a.id}`}
                  className="grid grid-cols-[1fr_auto_auto] items-center gap-6 py-4 transition-colors hover:bg-bg-1"
                >
                  <div>
                    <div
                      className="text-base font-medium"
                      style={{ letterSpacing: "-0.012em" }}
                    >
                      {fund?.name ?? "Unknown fund"}
                    </div>
                    <div className="mt-0.5 text-xs" style={{ color: "var(--ink-2)" }}>
                      {fund
                        ? `${fund.n_observations} months · ${fmtRange(fund.inception, fund.last_observation)}`
                        : "—"}
                    </div>
                  </div>
                  <span className="eyebrow">
                    {new Date(a.created_at).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                    })}
                  </span>
                  <span
                    className="mono"
                    style={{ color: "var(--ink-2)", fontSize: 11 }}
                  >
                    →
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
