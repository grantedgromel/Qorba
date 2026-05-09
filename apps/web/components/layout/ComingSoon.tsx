import Link from "next/link";
import { ArrowRight } from "lucide-react";

interface ComingSoonProps {
  title: string;
  sprint: string;
  description: string;
}

export function ComingSoon({ title, sprint, description }: ComingSoonProps) {
  return (
    <div className="page-enter mx-auto max-w-2xl px-7 py-24">
      <p className="eyebrow">{sprint}</p>
      <h1
        className="mt-2 font-serif italic"
        style={{
          fontSize: 56,
          lineHeight: 0.95,
          letterSpacing: "-0.01em",
          color: "var(--ink-0)",
        }}
      >
        {title}
      </h1>
      <p
        className="mt-6 max-w-md"
        style={{ fontSize: 14, color: "var(--ink-1)", lineHeight: 1.55 }}
      >
        {description}
      </p>
      <Link
        href="/dashboard"
        className="mt-8 inline-flex items-center gap-1.5 text-xs"
        style={{ color: "var(--ink-2)" }}
      >
        Back to overview <ArrowRight size={12} strokeWidth={1.5} />
      </Link>
    </div>
  );
}
