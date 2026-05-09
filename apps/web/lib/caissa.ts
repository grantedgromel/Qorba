"use client";

const KEY = "qorba.caissa.token";

export const caissaToken = {
  get(): string | null {
    if (typeof window === "undefined") return null;
    return sessionStorage.getItem(KEY);
  },
  set(token: string): void {
    if (typeof window === "undefined") return;
    sessionStorage.setItem(KEY, token);
  },
  clear(): void {
    if (typeof window === "undefined") return;
    sessionStorage.removeItem(KEY);
  },
};

export function caissaHeaders(): HeadersInit {
  const t = caissaToken.get();
  return t ? { "X-Caissa-Token": t } : {};
}

export async function caissaFetch(path: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers);
  const t = caissaToken.get();
  if (t) headers.set("X-Caissa-Token", t);
  return fetch(path, { ...init, headers, credentials: "include" });
}
