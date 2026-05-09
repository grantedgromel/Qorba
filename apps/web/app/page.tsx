import Link from "next/link";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-10 px-6 py-24">
      <div className="flex items-center gap-3">
        <div
          className="grid h-10 w-10 place-items-center rounded-[10px] text-base font-bold tracking-tighter"
          style={{ background: "var(--accent)", color: "var(--accent-ink)" }}
        >
          Q
        </div>
        <h1
          className="font-serif italic"
          style={{ fontSize: 48, lineHeight: 1, letterSpacing: "-0.02em" }}
        >
          Qorba
        </h1>
      </div>
      <p
        className="max-w-md text-center"
        style={{ fontSize: 14, color: "var(--ink-1)", lineHeight: 1.55 }}
      >
        Hedge-fund analytics for institutional allocators. Drop a tearsheet —
        get a Robinhood-grade view in seconds.
      </p>
      <div className="flex gap-3">
        <Link
          href="/register"
          className="inline-flex h-[30px] items-center gap-2 rounded-md px-3 text-xs font-medium"
          style={{ background: "var(--accent)", color: "var(--accent-ink)" }}
        >
          Get started
        </Link>
        <Link
          href="/login"
          className="inline-flex h-[30px] items-center gap-2 rounded-md border px-3 text-xs font-medium"
          style={{ borderColor: "var(--line)", color: "var(--ink-0)" }}
        >
          Sign in
        </Link>
      </div>
    </main>
  );
}
