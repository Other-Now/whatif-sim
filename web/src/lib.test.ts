import { describe, expect, it } from "vitest";
import { bands, diagramNodes, formatValue, reinforcingEdges } from "./lib";

const fan = { p05: [1, 2], p25: [2, 3], p50: [3, 4], p75: [4, 5], p95: [5, 6], mean: [3, 4] };

describe("bands", () => {
  it("pairs quantiles into outer and inner bands", () => {
    const b = bands([0, 1], fan);
    expect(b.outer).toEqual([{ t: 0, lo: 1, hi: 5 }, { t: 1, lo: 2, hi: 6 }]);
    expect(b.inner[1]).toEqual({ t: 1, lo: 3, hi: 5 });
    expect(b.median.map((m) => m.v)).toEqual([3, 4]);
  });
  it("rejects mismatched lengths", () => {
    expect(() => bands([0], fan)).toThrow(/length mismatch/);
  });
});

describe("diagram", () => {
  const model = { stocks: [{ name: "U" }], flows: [{ name: "hires" }] };
  const edges = [
    { from: "U", to: "hires", sign: 1 },
    { from: "hires", to: "U", sign: -1 },
    { from: "u", to: "hires", sign: -1 },
  ];
  it("types nodes from the model", () => {
    expect(diagramNodes(edges, model)).toEqual([
      { id: "U", kind: "stock" },
      { id: "hires", kind: "flow" },
      { id: "u", kind: "aux" },
    ]);
  });
  it("marks edges on reinforcing loops only", () => {
    const s = reinforcingEdges([
      { nodes: ["a", "b"], polarity: "R" },
      { nodes: ["c", "d"], polarity: "B" },
    ]);
    expect([...s].sort()).toEqual(["a->b", "b->a"]);
  });
});

it("formats dimensionless rates as percent", () => {
  expect(formatValue(0.0412, "1")).toBe("4.12%");
  expect(formatValue(7123, "kpeople")).toBe((7123).toLocaleString(undefined, { maximumSignificantDigits: 4 }));
});
