"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Dropzone } from "@/components/ingest/Dropzone";
import { Segmented } from "@/components/ui/segmented";
import { Button } from "@/components/ui/button";

type Mode = "file" | "paste";

export default function NewAnalysisPage() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("file");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pasted, setPasted] = useState("");
  const [pasteName, setPasteName] = useState("Manager");

  async function postIngest(path: string, init: RequestInit) {
    setError(null);
    setBusy(true);
    try {
      const r = await fetch(path, { ...init, credentials: "include" });
      if (!r.ok) {
        if (r.status === 401) {
          router.push("/login");
          return;
        }
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail ?? "Ingest failed");
      }
      const draft = await r.json();
      router.push(`/analyses/new/confirm/${draft.id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  function endpointFor(filename: string): string {
    const lower = filename.toLowerCase();
    if (lower.endsWith(".pdf")) return "/api/v1/ingest/pdf";
    if (lower.endsWith(".xlsx") || lower.endsWith(".xls")) return "/api/v1/ingest/xlsx";
    return "/api/v1/ingest/csv";
  }

  async function handleFile(file: File) {
    const form = new FormData();
    form.append("file", file);
    await postIngest(endpointFor(file.name), { method: "POST", body: form });
  }

  async function handlePaste() {
    if (!pasted.trim()) {
      setError("Paste a date+return table first.");
      return;
    }
    await postIngest("/api/v1/ingest/paste", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: pasteName || "Manager", text: pasted }),
    });
  }

  return (
    <div className="container max-w-2xl py-16">
      <div className="mb-8 space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">New analysis</h1>
        <p className="text-sm text-muted">
          Drop a CSV, Excel, or PDF tearsheet — or paste a table. You&apos;ll
          confirm what got parsed before it&apos;s saved.
        </p>
      </div>

      <Segmented<Mode>
        value={mode}
        onChange={setMode}
        options={[
          { value: "file", label: "Upload" },
          { value: "paste", label: "Paste" },
        ]}
        className="mb-4"
      />

      <Card>
        <CardContent className="space-y-4 p-6">
          {mode === "file" ? (
            <Dropzone
              onFile={handleFile}
              disabled={busy}
              accept=".csv,.xlsx,.xls,.pdf"
              hint="CSV · XLSX · PDF tearsheet"
            />
          ) : (
            <div className="space-y-3">
              <input
                value={pasteName}
                onChange={(e) => setPasteName(e.target.value)}
                placeholder="Manager name"
                className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-fg placeholder:text-muted focus:outline-none focus:ring-1 focus:ring-accent"
                disabled={busy}
              />
              <textarea
                value={pasted}
                onChange={(e) => setPasted(e.target.value)}
                rows={10}
                placeholder={`date,return\n2024-01-31,1.23\n2024-02-29,-0.45`}
                className="w-full rounded-md border border-border bg-surface px-3 py-2 font-mono text-sm text-fg placeholder:text-muted focus:outline-none focus:ring-1 focus:ring-accent"
                disabled={busy}
              />
              <Button onClick={handlePaste} disabled={busy}>
                {busy ? "Parsing…" : "Parse"}
              </Button>
            </div>
          )}
          {busy && mode === "file" && (
            <p className="text-sm text-muted">Parsing…</p>
          )}
          {error && <p className="text-sm text-negative">{error}</p>}
        </CardContent>
      </Card>
    </div>
  );
}
