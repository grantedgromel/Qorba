"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Dropzone } from "@/components/ingest/Dropzone";
import { Card, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api/client";

export default function NewAnalysisPage() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stage, setStage] = useState<string>("");

  async function handleFile(file: File) {
    setError(null);
    setBusy(true);

    try {
      // 1. Ingest CSV
      setStage("Parsing CSV…");
      const form = new FormData();
      form.append("file", file);
      const ingestResp = await fetch("/api/v1/ingest/csv", {
        method: "POST",
        credentials: "include",
        body: form,
      });
      if (!ingestResp.ok) {
        if (ingestResp.status === 401) {
          router.push("/login");
          return;
        }
        const body = await ingestResp.json().catch(() => ({}));
        throw new Error(body.detail ?? "Could not parse CSV");
      }
      const extracted = await ingestResp.json();

      // Sprint 1 shortcut: skip the correction UI, persist immediately.
      // Sprint 2 inserts <ReturnGrid> here so the user confirms pct/decimal.

      // 2. Persist as a fund
      setStage("Saving fund…");
      const { data: fund, error: fundErr } = await api.POST("/api/v1/funds", {
        body: extracted,
      });
      if (fundErr || !fund) throw new Error("Could not save fund");

      // 3. Create analysis with the default Sprint 1 metric set
      setStage("Creating analysis…");
      const { data: analysis, error: anaErr } = await api.POST("/api/v1/analyses", {
        body: {
          fund_id: fund.id,
          metrics: {
            metric_ids: ["sharpe", "ann_return_geo", "ann_vol", "max_dd", "win_rate", "sortino"],
          },
          rf_annual: 0.0,
          mar_annual: 0.0,
          omega_threshold: 0.0,
        },
      });
      if (anaErr || !analysis) throw new Error("Could not create analysis");

      router.push(`/analyses/${analysis.id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
      setStage("");
    }
  }

  return (
    <div className="container max-w-2xl py-16">
      <div className="mb-8 space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">New analysis</h1>
        <p className="text-sm text-muted">
          Drop a CSV with a date column and a monthly return column. The first
          numeric column is treated as the fund return series.
        </p>
      </div>
      <Card>
        <CardContent className="p-6">
          <Dropzone onFile={handleFile} disabled={busy} hint="CSV with date + return columns" />
          {busy && <p className="mt-4 text-sm text-muted">{stage}</p>}
          {error && <p className="mt-4 text-sm text-negative">{error}</p>}
        </CardContent>
      </Card>
    </div>
  );
}
