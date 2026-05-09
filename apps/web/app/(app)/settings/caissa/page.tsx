"use client";

import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { caissaFetch, caissaToken } from "@/lib/caissa";

interface Status {
  connected: boolean;
  expires_in_seconds: number | null;
  tenant_id: string | null;
  user_email: string | null;
  scopes: string[];
}

export default function CaissaSettingsPage() {
  const [pasted, setPasted] = useState("");
  const [status, setStatus] = useState<Status | null>(null);
  const [probeBody, setProbeBody] = useState<string>("");
  const [probePath, setProbePath] = useState("/api/v1/benchmarks");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function refreshStatus() {
    const r = await caissaFetch("/api/v1/integrations/caissa/status");
    if (r.ok) setStatus(await r.json());
  }

  useEffect(() => {
    setPasted(caissaToken.get() ?? "");
    void refreshStatus();
  }, []);

  function save() {
    if (!pasted.trim()) {
      setError("Paste a token first.");
      return;
    }
    caissaToken.set(pasted.trim());
    setError(null);
    void refreshStatus();
  }

  function clear() {
    caissaToken.clear();
    setPasted("");
    setStatus(null);
    setProbeBody("");
  }

  async function probe() {
    setBusy(true);
    setError(null);
    setProbeBody("");
    try {
      const r = await caissaFetch(
        `/api/v1/integrations/caissa/probe?path=${encodeURIComponent(probePath)}`,
      );
      const body = await r.json();
      setProbeBody(JSON.stringify(body, null, 2));
      if (!r.ok) setError(body.detail ?? "Probe failed");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Probe failed");
    } finally {
      setBusy(false);
    }
  }

  const expiresInMin =
    status?.expires_in_seconds != null
      ? Math.floor(status.expires_in_seconds / 60)
      : null;

  return (
    <div className="container max-w-3xl space-y-6 py-12">
      <div className="space-y-1">
        <p className="text-xs uppercase tracking-wider text-muted">Integration</p>
        <h1 className="text-2xl font-semibold tracking-tight">Caissa</h1>
        <p className="text-sm text-muted">
          Bridge mode: paste a Caissa access token from the Swagger UI. Qorba
          uses it to read benchmarks via your tenant. Tokens expire ~6 minutes
          after they&apos;re issued.
        </p>
      </div>

      <Card>
        <CardContent className="space-y-3 p-6">
          <ol className="space-y-1 text-sm text-muted">
            <li>
              1. Open{" "}
              <a
                className="text-accent hover:underline"
                href="https://client-api.caissallc.com/index.html?urls.primaryName=Total%20Plan%20Manager%20Client%20API%20v1"
                target="_blank"
                rel="noreferrer"
              >
                Caissa Swagger UI
              </a>{" "}
              and click &quot;Authorize.&quot;
            </li>
            <li>2. After you sign in, copy the access_token from the URL fragment.</li>
            <li>3. Paste it here.</li>
          </ol>
          <textarea
            value={pasted}
            onChange={(e) => setPasted(e.target.value)}
            rows={4}
            placeholder="eyJhbGciOiJSUzI1NiIs…"
            className="w-full rounded-md border border-border bg-surface px-3 py-2 font-mono text-xs text-fg placeholder:text-muted focus:outline-none focus:ring-1 focus:ring-accent"
          />
          <div className="flex items-center gap-2">
            <Button onClick={save}>Save token</Button>
            <Button variant="ghost" onClick={clear}>
              Clear
            </Button>
          </div>
          {error && <p className="text-sm text-negative">{error}</p>}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="space-y-3 p-6">
          <h2 className="text-sm font-semibold">Status</h2>
          {!status || !status.connected ? (
            <p className="text-sm text-muted">Not connected.</p>
          ) : (
            <dl className="grid grid-cols-2 gap-2 text-sm">
              <dt className="text-muted">Tenant</dt>
              <dd className="font-mono text-fg">{status.tenant_id ?? "—"}</dd>
              <dt className="text-muted">User</dt>
              <dd className="font-mono text-fg">{status.user_email ?? "—"}</dd>
              <dt className="text-muted">Expires in</dt>
              <dd className="font-mono text-fg">{expiresInMin}m</dd>
              <dt className="text-muted">Scopes</dt>
              <dd className="font-mono text-fg">{status.scopes.join(" ")}</dd>
            </dl>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="space-y-3 p-6">
          <h2 className="text-sm font-semibold">Probe an endpoint</h2>
          <p className="text-xs text-muted">
            Caissa endpoint paths aren&apos;t finalised. Try one from the
            Swagger UI here to confirm it works with your token before we
            wire it up properly.
          </p>
          <div className="flex items-center gap-2">
            <input
              value={probePath}
              onChange={(e) => setProbePath(e.target.value)}
              className="flex-1 rounded-md border border-border bg-surface px-3 py-2 font-mono text-xs text-fg focus:outline-none focus:ring-1 focus:ring-accent"
            />
            <Button onClick={probe} disabled={busy || !status?.connected}>
              {busy ? "Probing…" : "Probe"}
            </Button>
          </div>
          {probeBody && (
            <pre className="max-h-64 overflow-auto rounded-md border border-border bg-elevated p-3 font-mono text-xs text-fg">
              {probeBody}
            </pre>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
