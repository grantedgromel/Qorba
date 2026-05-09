"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutGrid,
  TrendingUp,
  AlertTriangle,
  GitBranch,
  Sigma,
  LineChart,
  Users,
  Activity,
  Calendar,
  Upload,
  FileSpreadsheet,
  Plug,
  type LucideIcon,
} from "lucide-react";

interface NavItem {
  href: string;
  label: string;
  Icon: LucideIcon;
  enabled: boolean;
}

const TOP: NavItem[] = [
  { href: "/dashboard", label: "Overview", Icon: LayoutGrid, enabled: true },
  { href: "/returns", label: "Return Measures", Icon: TrendingUp, enabled: false },
  { href: "/risk", label: "Risk Measures", Icon: AlertTriangle, enabled: false },
  { href: "/regression", label: "Regression", Icon: GitBranch, enabled: false },
  { href: "/factors", label: "Factor Analysis", Icon: Sigma, enabled: false },
  { href: "/rolling", label: "Rolling", Icon: LineChart, enabled: false },
  { href: "/peer", label: "Peer Group", Icon: Users, enabled: false },
  { href: "/drawdown", label: "Drawdown", Icon: Activity, enabled: false },
  { href: "/calendar", label: "Calendar", Icon: Calendar, enabled: false },
];

const BOTTOM: NavItem[] = [
  { href: "/benchmarks/import", label: "Benchmarks", Icon: FileSpreadsheet, enabled: true },
  { href: "/settings/caissa", label: "Caissa", Icon: Plug, enabled: true },
  { href: "/analyses/new", label: "Upload", Icon: Upload, enabled: true },
];

function isActive(pathname: string, href: string): boolean {
  if (href === "/dashboard") {
    return pathname === "/dashboard" || pathname.startsWith("/analyses/");
  }
  return pathname === href || pathname.startsWith(href + "/");
}

export function Rail() {
  const pathname = usePathname();

  const renderItem = (item: NavItem) => {
    const active = isActive(pathname, item.href);
    const className = [
      "relative grid h-9 w-9 place-items-center rounded-md transition-colors",
      active ? "bg-bg-2 text-ink-0" : "text-ink-2",
      item.enabled ? "hover:text-ink-0 hover:bg-bg-2" : "opacity-30 cursor-not-allowed",
    ].join(" ");

    const inner = (
      <>
        <item.Icon size={16} strokeWidth={1.5} />
        {active && (
          <span
            aria-hidden
            className="absolute top-2 bottom-2 w-0.5 rounded-full bg-accent"
            style={{ left: -10 }}
          />
        )}
      </>
    );

    if (!item.enabled) {
      return (
        <span key={item.href} title={`${item.label} · coming soon`} className={className}>
          {inner}
        </span>
      );
    }
    return (
      <Link key={item.href} href={item.href} title={item.label} className={className}>
        {inner}
      </Link>
    );
  };

  return (
    <aside
      className="flex h-full w-14 flex-shrink-0 flex-col items-center gap-1 border-r py-3.5"
      style={{ background: "var(--bg-0)", borderColor: "var(--line)" }}
    >
      <Link
        href="/dashboard"
        aria-label="Qorba"
        className="mb-3.5 grid h-8 w-8 place-items-center rounded-[9px] text-[15px] font-bold tracking-tighter"
        style={{ background: "var(--accent)", color: "var(--accent-ink)" }}
      >
        Q
      </Link>
      {TOP.map(renderItem)}
      <div className="flex-1" />
      {BOTTOM.map(renderItem)}
    </aside>
  );
}
