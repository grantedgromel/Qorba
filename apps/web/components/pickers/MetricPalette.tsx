"use client";

import { Command } from "cmdk";
import { useEffect, useMemo, useState } from "react";
import type { MetricCatalogEntry } from "@/app/(app)/analyses/[id]/page";

interface MetricPaletteProps {
  catalog: MetricCatalogEntry[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  onResetDefaults: () => void;
  onClearAll: () => void;
}

export function MetricPalette({
  catalog,
  selected,
  onToggle,
  onResetDefaults,
  onClearAll,
}: MetricPaletteProps) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      } else if (e.key === "Escape" && open) {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const grouped = useMemo(() => {
    const g: Record<string, MetricCatalogEntry[]> = {};
    for (const item of catalog) {
      g[item.group] ??= [];
      g[item.group]!.push(item);
    }
    return g;
  }, [catalog]);

  const groupOrder = [
    "Returns",
    "Risk",
    "Drawdown",
    "Risk-Adjusted",
    "Distributional",
    "Benchmark-Relative",
  ];

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex h-7 items-center gap-1.5 rounded-md border px-2 text-[11px] font-medium transition-colors"
        style={{
          borderColor: "var(--line)",
          color: "var(--ink-1)",
        }}
      >
        + Edit metrics
        <span
          className="mono rounded-sm border px-1 text-[9.5px]"
          style={{ borderColor: "var(--line)", color: "var(--ink-2)" }}
        >
          ⌘K
        </span>
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center pt-[18vh]"
          style={{ background: "color-mix(in oklab, black 50%, transparent)" }}
          onClick={() => setOpen(false)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="card w-full max-w-xl overflow-hidden"
            style={{
              boxShadow: "0 20px 60px rgba(0,0,0,0.4)",
              maxHeight: "70vh",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <Command label="Metric palette" loop>
              <div
                className="flex items-center gap-2 px-4 py-3"
                style={{ borderBottom: "1px solid var(--line)" }}
              >
                <span
                  className="mono"
                  style={{ fontSize: 10, letterSpacing: "0.12em", color: "var(--ink-2)" }}
                >
                  ⌘K
                </span>
                <Command.Input
                  placeholder="Search metrics, profiles, actions…"
                  className="flex-1 bg-transparent text-sm outline-none"
                  style={{ color: "var(--ink-0)" }}
                  autoFocus
                />
              </div>
              <Command.List
                style={{
                  overflowY: "auto",
                  padding: "8px 0",
                }}
              >
                <Command.Empty>
                  <div
                    style={{
                      padding: "20px 16px",
                      fontSize: 12,
                      color: "var(--ink-2)",
                    }}
                  >
                    No matches.
                  </div>
                </Command.Empty>
                <Command.Group
                  heading="Actions"
                  className="qorba-cmd-group"
                >
                  <Command.Item
                    onSelect={() => {
                      onResetDefaults();
                      setOpen(false);
                    }}
                  >
                    <span style={{ flex: 1 }}>Reset to default profile</span>
                    <span className="mono" style={{ fontSize: 10, color: "var(--ink-2)" }}>
                      ⌘R
                    </span>
                  </Command.Item>
                  <Command.Item
                    onSelect={() => {
                      onClearAll();
                      setOpen(false);
                    }}
                  >
                    <span style={{ flex: 1 }}>Clear all</span>
                  </Command.Item>
                </Command.Group>
                {groupOrder
                  .filter((g) => grouped[g])
                  .map((group) => (
                    <Command.Group key={group} heading={group} className="qorba-cmd-group">
                      {grouped[group]!.map((item) => {
                        const isOn = selected.has(item.id);
                        return (
                          <Command.Item
                            key={item.id}
                            value={`${item.label} ${item.id}`}
                            onSelect={() => onToggle(item.id)}
                          >
                            <span
                              aria-hidden
                              className="grid h-3.5 w-3.5 place-items-center rounded-sm border"
                              style={{
                                borderColor: isOn ? "var(--accent)" : "var(--line)",
                                background: isOn ? "var(--accent)" : "transparent",
                                color: "var(--accent-ink)",
                                fontSize: 9,
                              }}
                            >
                              {isOn ? "✓" : ""}
                            </span>
                            <span style={{ flex: 1 }}>{item.label}</span>
                            {item.requires_benchmark && (
                              <span
                                className="mono"
                                style={{ fontSize: 9.5, color: "var(--ink-2)" }}
                              >
                                BMK
                              </span>
                            )}
                            <span
                              className="mono"
                              style={{ fontSize: 9.5, color: "var(--ink-2)" }}
                            >
                              {item.id}
                            </span>
                          </Command.Item>
                        );
                      })}
                    </Command.Group>
                  ))}
              </Command.List>
            </Command>
          </div>
        </div>
      )}

      <style jsx global>{`
        .qorba-cmd-group [cmdk-group-heading] {
          padding: 8px 16px 4px;
          font-family: var(--font-mono);
          font-size: 10px;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: var(--ink-2);
          font-weight: 500;
        }
        [cmdk-item] {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 16px;
          font-size: 13px;
          color: var(--ink-0);
          cursor: pointer;
          transition: background 100ms;
        }
        [cmdk-item][data-selected="true"] {
          background: var(--bg-2);
        }
      `}</style>
    </>
  );
}
