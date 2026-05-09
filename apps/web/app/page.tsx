import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <main className="container flex min-h-screen flex-col items-center justify-center gap-8 py-24">
      <div className="space-y-3 text-center">
        <h1 className="text-5xl font-semibold tracking-tight">Qorba</h1>
        <p className="text-lg text-muted">
          Public-manager analytics for institutional allocators.
        </p>
      </div>
      <div className="flex gap-3">
        <Button asChild>
          <Link href="/register">Get started</Link>
        </Button>
        <Button asChild variant="secondary">
          <Link href="/login">Sign in</Link>
        </Button>
      </div>
    </main>
  );
}
