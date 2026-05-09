"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api/client";

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    const { error: err } = await api.POST("/api/v1/auth/register", {
      body: { email, password },
    });
    setBusy(false);
    if (err) {
      setError(typeof err.detail === "string" ? err.detail : "Registration failed");
      return;
    }
    router.push("/dashboard");
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-6 py-24">
      <form onSubmit={onSubmit} className="card w-full max-w-sm space-y-4 p-6">
        <div>
          <h1 className="text-lg font-medium" style={{ letterSpacing: "-0.012em" }}>
            Create your account
          </h1>
          <p className="mt-1 text-xs" style={{ color: "var(--ink-2)" }}>
            Single-user mode. Pick something you&apos;ll remember.
          </p>
        </div>
        <div className="space-y-2">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            required
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="password">Password</Label>
          <Input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="new-password"
            required
            minLength={12}
          />
        </div>
        {error && (
          <p className="text-xs" style={{ color: "var(--neg)" }}>
            {error}
          </p>
        )}
        <Button type="submit" className="w-full" disabled={busy}>
          {busy ? "Creating…" : "Create account"}
        </Button>
        <p className="text-center text-xs" style={{ color: "var(--ink-2)" }}>
          Already have one?{" "}
          <Link href="/login" className="text-accent hover:underline">
            Sign in
          </Link>
        </p>
      </form>
    </main>
  );
}
