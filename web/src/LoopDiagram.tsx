import * as d3 from "d3";
import { useEffect, useRef } from "react";
import { diagramNodes, reinforcingEdges, type Report } from "./lib";

type Model = { stocks: { name: string }[]; flows: { name: string }[] };
type N = d3.SimulationNodeDatum & { id: string; kind: string };
type L = d3.SimulationLinkDatum<N> & { sign: number; key: string };

const W = 560;
const H = 360;

/** Causal-loop diagram: force layout, + / - link signs, reinforcing loops in red. */
export function LoopDiagram({ report, model }: { report: Report; model: Model }) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(ref.current!);
    svg.selectAll("*").remove();
    const nodes: N[] = diagramNodes(report.edges, model).map((n) => ({ ...n }));
    const links: L[] = report.edges.map((e) => ({ source: e.from, target: e.to, sign: e.sign, key: `${e.from}->${e.to}` }));
    const hot = reinforcingEdges(report.loops);

    svg.append("defs").selectAll("marker").data(["plain", "hot"]).join("marker")
      .attr("id", (d) => `arrow-${d}`).attr("viewBox", "0 -5 10 10").attr("refX", 24).attr("refY", 0)
      .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
      .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", (d) => (d === "hot" ? "var(--hot)" : "var(--muted)"));

    const link = svg.append("g").selectAll("g").data(links).join("g");
    const line = link.append("line")
      .attr("stroke", (d) => (hot.has(d.key) ? "var(--hot)" : "var(--muted)"))
      .attr("stroke-width", (d) => (hot.has(d.key) ? 2 : 1.2))
      .attr("marker-end", (d) => `url(#arrow-${hot.has(d.key) ? "hot" : "plain"})`);
    const sign = link.append("text").attr("class", "sign")
      .text((d) => (d.sign > 0 ? "+" : d.sign < 0 ? "−" : "?"));

    const node = svg.append("g").selectAll("g").data(nodes).join("g").attr("class", (d) => `node ${d.kind}`);
    node.append("rect").attr("x", -44).attr("y", -13).attr("width", 88).attr("height", 26)
      .attr("rx", (d) => (d.kind === "stock" ? 2 : 13));
    node.append("text").attr("text-anchor", "middle").attr("dy", 4).text((d) => d.id.replace(/_/g, " "));

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink<N, L>(links).id((d) => d.id).distance(110))
      .force("charge", d3.forceManyBody().strength(-500))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collide", d3.forceCollide(50));
    sim.on("tick", () => {
      for (const n of nodes) {
        n.x = Math.max(50, Math.min(W - 50, n.x!));
        n.y = Math.max(20, Math.min(H - 20, n.y!));
      }
      line.attr("x1", (d) => (d.source as N).x!).attr("y1", (d) => (d.source as N).y!)
        .attr("x2", (d) => (d.target as N).x!).attr("y2", (d) => (d.target as N).y!);
      sign.attr("x", (d) => ((d.source as N).x! * 0.4 + (d.target as N).x! * 0.6))
        .attr("y", (d) => ((d.source as N).y! * 0.4 + (d.target as N).y! * 0.6) - 4);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });
    return () => void sim.stop();
  }, [report, model]);

  return <svg ref={ref} viewBox={`0 0 ${W} ${H}`} className="diagram" role="img" aria-label="causal loop diagram" />;
}
