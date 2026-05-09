"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { caissaFetch, caissaToken } from "@/lib/caissa";

interface Item {
  code: string;
  name?: string;
  category?: string;
}

type ImportState = "idle" | "loading" | "saved" | "error";

export default function CaissaImportPage() {
  const [items, setItems] = useState<Item[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [state, setState] = useState<Record<string, ImportState>>({});
  const [hasToken, setHasToken] = useState(false);

  useEffect(() => {
    setHasToken(!!caissaToken.get());
    void load();
  }, []);

  async function load() {
    setBusy(true);
    setError(null);
    try {
      const r = await caissaFetch("/api/v1/integrations/caissa/list-benchmarks");
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${r.status}`);
      }
      const body = await r.json();
      setItems(body.items as Item[]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not list benchmarks");
    } finally {
      setBusy(false);
    }
  }

  async function importOne(item: Item) {
    setState((s) => ({ ...s, [item.code]: "loading" }));
    try {
      const r = await caissaFetch("/api/v1/integrations/caissa/import-benchmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: item.code, name: item.name ?? item.code }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setState((s) => ({ ...s, [item.code]: "saved" }));
    } catch {
      setState((s) => ({ ...s, [item.code]: "error" }));
    }
  }

  return (
    <div className="container max-w-3xl space-y-6 py-12">
      <div className="space-y-1">
        <p className="text-xs uppercase tracking-wider text-ink-2">Benchmarks</p>
        <h1 className="text-2xl font-semibold tracking-tight">Import from Caissa</h1>
        <p className="text-sm text-ink-2">
          Pick benchmarks from your Caissa tenant. Each import persists the
          monthly returns into Qorba so you don&apos;t need a fresh token next
          time.
        </p>
      </div>

      {!hasToken && (
        <Card>
          <CardContent className="space-y-3 p-6">
            <p className="text-sm text-ink-0">No Caissa token in this session.</p>
            <Button asChild>
              <Link href="/settings/caissa">Connect Caissa</Link>
            </Button>
          </CardContent>
        </Card>
      )}

      {hasToken && (
        <Card>
          <CardContent className="space-y-3 p-6">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold">Available</h2>
              <Button variant="ghost" onClick={load} disabled={busy}>
                {busy ? "Loading…" : "Refresh"}
              </Button>
            </div>
            {error && <p className="text-sm text-neg">{error}</p>}
            {items.length === 0 && !busy && !error && (
              <p className="text-sm text-ink-2">
                No benchmarks listed. Endpoint paths inside Caissa aren&apos;t
                finalised — try the probe at{" "}
                <Link href="/settings/caissa" className="text-accent hover:underline">
                  /settings/caissa
                </Link>{" "}
                to discover the right path.
              </p>
            )}
            <ul className="divide-y divide-line">
              {items.map((item) => {
                const st = state[item.code] ?? "idle";
                return (
                  <li key={item.code} className="flex items-center justify-between py-3">
                    <div>
                      <div className="font-medium text-ink-0">{item.name ?? item.code}</div>
                      <div className="font-mono text-xs text-ink-2">
                        {item.code}
                        {item.category ? ` · ${item.category}` : ""}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      variant={st === "saved" ? "secondary" : "default"}
                      onClick={() => importOne(item)}
                      disabled={st === "loading" || st === "saved"}
                    >
                      {st === "loading"
                        ? "Importing…"
                        : st === "saved"
                          ? "Imported"
                          : st === "error"
                            ? "Retry"
                            : "Import"}
                    </Button>
                  </li>
                );
              })}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
