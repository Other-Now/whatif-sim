// Pure helpers, kept out of the components so they can be unit tested.

export type Fan = { p05: number[]; p25: number[]; p50: number[]; p75: number[]; p95: number[]; mean: number[] };

export type Edge = { from: string; to: string; sign: number };
export type Loop = { nodes: string[]; polarity: "R" | "B" | "?" };
export type Issue = { code: string; where: string; message: string; severity: "error" | "warning" };
export type Report = { ok: boolean; issues: Issue[]; loops: Loop[]; edges: Edge[] };

export type Band = { t: number; lo: number; hi: number };

/** Turn a fan (quantile arrays over time) into the two bands a chart draws. */
export function bands(t: number[], f: Fan): { outer: Band[]; inner: Band[]; median: { t: number; v: number }[] } {
  if (t.length !== f.p50.length) throw new Error(`length mismatch: ${t.length} times, ${f.p50.length} values`);
  return {
    outer: t.map((x, i) => ({ t: x, lo: f.p05[i], hi: f.p95[i] })),
    inner: t.map((x, i) => ({ t: x, lo: f.p25[i], hi: f.p75[i] })),
    median: t.map((x, i) => ({ t: x, v: f.p50[i] })),
  };
}

export type NodeKind = "stock" | "flow" | "aux";

/** Nodes for the causal-loop diagram: every node in an edge, typed by the model. */
export function diagramNodes(
  edges: Edge[],
  model: { stocks: { name: string }[]; flows: { name: string }[] },
): { id: string; kind: NodeKind }[] {
  const stocks = new Set(model.stocks.map((s) => s.name));
  const flows = new Set(model.flows.map((f) => f.name));
  const ids = [...new Set(edges.flatMap((e) => [e.from, e.to]))].sort();
  return ids.map((id) => ({ id, kind: stocks.has(id) ? "stock" : flows.has(id) ? "flow" : "aux" }));
}

/** Edges that lie on at least one reinforcing loop, for highlighting. */
export function reinforcingEdges(loops: Loop[]): Set<string> {
  const out = new Set<string>();
  for (const l of loops) {
    if (l.polarity !== "R") continue;
    l.nodes.forEach((n, i) => out.add(`${n}->${l.nodes[(i + 1) % l.nodes.length]}`));
  }
  return out;
}

export function formatValue(v: number, unit: string): string {
  if (unit === "1" && Math.abs(v) < 1) return `${(100 * v).toFixed(2)}%`;
  return v.toLocaleString(undefined, { maximumSignificantDigits: 4 });
}
