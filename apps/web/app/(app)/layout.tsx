import Link from "next/link";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="flex h-14 items-center justify-between border-b border-border px-6">
        <Link href="/analyses/new" className="text-sm font-semibold tracking-tight">
          Qorba
        </Link>
        <nav className="flex items-center gap-4 text-sm text-muted">
          <Link href="/analyses/new" className="hover:text-fg">
            New analysis
          </Link>
          <Link href="/benchmarks/import" className="hover:text-fg">
            Benchmarks
          </Link>
          <Link href="/settings/caissa" className="hover:text-fg">
            Caissa
          </Link>
        </nav>
      </header>
      <main className="flex-1">{children}</main>
    </div>
  );
}
