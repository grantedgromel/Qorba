"use client";

import { use, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Segmented } from "@/components/ui/segmented";
import { Sparkline } from "@/components/charts/Sparkline";
import {
  applyEdits,
  CellEdit,
  ReturnGrid,
  ReturnPoint,
} from "@/components/ingest/ReturnGrid";

type Scale = "percent" | "decimal";

interface Draft {
  id: string;
  name: string;
  detected_scale: Scale;
  tier_used: 1 | 2 | 3 | 4;
  confidence: number;
  confidence_components?: Record<string, number>;
  points: ReturnPoint[];
  warnings: string[];
}

export default function ConfirmDraftPage({
  params,
}: {
  params: Promise<{ draftId: string }>;
}) {
  const { draftId } = use(params);
  const router = useRouter();

  const [draft, setDraft] = useState<Draft | null>(null);
  const [scale, setScale] = useState<Scale>("decimal");
  const [name, setName] = useState("");
  const [edits, setEdits] = useState<Record<string, CellEdit>>({});
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const r = await fetch(`/api/v1/ingest/drafts/${draftId}`, {
        credentials: "include",
      });
      if (r.status === 401) {
        router.push("/login");
        return;
      }
      if (!r.ok) {
        setError("Could not load this ingestion draft.");
        return;
      }
      const d: Draft = await r.json();
      if (!cancelled) {
        setDraft(d);
        setScale(d.detected_scale);
        setName(d.name);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [draftId, router]);

  function handleEdit(period: string, raw: string) {
    if (!draft) return;
    setEdits((prev) => {
      const next = { ...prev };
      if (raw.trim() === "") {
        next[period] = { period, value: null };
        return next;
      }
      const num = Number(raw);
      if (!Number.isFinite(num)) return prev;
      next[period] = { period, value: num };
      return next;
    });
  }

  const decimalPreview = useMemo(() => {
    if (!draft) return [];
    const merged = applyEdits(draft.points, edits).filter(
      (p): p is { period: string; value: number } => p.value !== null,
    );
    const factor = scale === "percent" ? 100 : 1;
    return merged.map((p) => p.value / factor);
  }, [draft, edits, scale]);

  async function confirm() {
    if (!draft) return;
    setSubmitting(true);
    setError(null);
    const merged = applyEdits(draft.points, edits);
    const editArray: CellEdit[] = Object.values(edits).filter((e) => {
      const original = draft.points.find((p) => p.period === e.period);
      return original ? original.value !== e.value : true;
    });
    const r = await fetch(`/api/v1/ingest/drafts/${draftId}/confirm`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scale, name, edits: editArray }),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      setError(body.detail ?? "Could not save fund");
      setSubmitting(false);
      return;
    }
    const fund = await r.json();
    const a = await fetch("/api/v1/analyses", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fund_id: fund.id,
        metrics: {
          metric_ids: ["sharpe", "ann_return_geo", "ann_vol", "max_dd", "win_rate", "sortino"],
        },
        rf_annual: 0,
        mar_annual: 0,
        omega_threshold: 0,
      }),
    });
    if (!a.ok) {
      setError("Saved fund, but couldn't create the analysis.");
      setSubmitting(false);
      return;
    }
    const analysis = await a.json();
    router.push(`/analyses/${analysis.id}`);
    void merged;
  }

  if (error && !draft) {
    return (
      <div className="container max-w-3xl py-16">
        <p className="text-sm text-negative">{error}</p>
      </div>
    );
  }

  if (!draft) {
    return (
      <div className="container max-w-3xl py-16">
        <p className="text-sm text-muted">Loading parsed result…</p>
      </div>
    );
  }

  const tierLabel = {
    1: "PyMuPDF",
    2: "pdfplumber",
    3: "AI vision",
    4: "Manual paste",
  }[draft.tier_used];

  return (
    <div className="container max-w-5xl space-y-6 py-12">
      <div className="space-y-1">
        <p className="text-xs uppercase tracking-wider text-muted">
          Confirm parsed returns
        </p>
        <div className="flex items-center justify-between gap-4">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full max-w-md bg-transparent text-2xl font-semibold tracking-tight text-fg outline-none placeholder:text-muted"
            placeholder="Manager name"
          />
          <div className="text-right text-xs text-muted">
            <div>Parsed by {tierLabel}</div>
            <div>Confidence {(draft.confidence * 100).toFixed(0)}%</div>
          </div>
        </div>
      </div>

      <Card>
        <CardContent className="space-y-4 p-6">
          <Sparkline values={decimalPreview} />
          <div className="flex items-center gap-3 text-sm">
            <span className="text-muted">Values are:</span>
            <Segmented<Scale>
              value={scale}
              onChange={setScale}
              options={[
                { value: "percent", label: "Percent (%)" },
                { value: "decimal", label: "Decimal" },
              ]}
            />
            {scale !== draft.detected_scale && (
              <span className="text-xs text-muted">
                (we guessed {draft.detected_scale})
              </span>
            )}
          </div>
          {draft.warnings.length > 0 && (
            <ul className="space-y-1 text-xs text-muted">
              {draft.warnings.map((w, i) => (
                <li key={i}>• {w}</li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <ReturnGrid
        points={draft.points}
        edits={edits}
        onEdit={handleEdit}
        scale={scale}
      />

      {error && <p className="text-sm text-negative">{error}</p>}

      <div className="flex items-center justify-end gap-3">
        <Button variant="ghost" onClick={() => router.push("/analyses/new")}>
          Cancel
        </Button>
        <Button onClick={confirm} disabled={submitting}>
          {submitting ? "Saving…" : "Confirm and analyze"}
        </Button>
      </div>
    </div>
  );
}
