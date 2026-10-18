import { useEffect, useState } from "react";
import { FanChart } from "./FanChart";
import { LoopDiagram } from "./LoopDiagram";
import type { Fan, Report } from "./lib";

type Example = { id: string; name: string; question: string };
type Model = {
  name: string;
  question: string;
  time_unit: string;
  stocks: { name: string; unit: string }[];
  flows: { name: string }[];
  aux: { name: string; unit: string }[];
  outputs: string[];
  scenarios: { name: string; description?: string }[];
};
type Status = {
  status: string;
  prefix_done: number;
  stop_batch: number | null;
  retried_claims: number;
  convergence: { mean: { est: number; halfwidth: number }; p95: { halfwidth: number } } | null;
};
type Result = {
  t: number[];
  samples: number;
  batches: number;
  baseline: Record<string, Fan>;
  scenario?: Record<string, Fan>;
  effect?: Record<string, Fan>;
};

async function api<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, body === undefined ? undefined : {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
  const text = await r.text();
  if (!r.ok) throw new Error(`${r.status}: ${text}`);
  return (r.headers.get("content-type") ?? "").includes("json") ? JSON.parse(text) : (text as T);
}

export function App() {
  const [examples, setExamples] = useState<Example[]>([]);
  const [model, setModel] = useState<Model | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [brief, setBrief] = useState("");
  const [question, setQuestion] = useState("");
  const [error, setError] = useState("");

  useEffect(() => { api<Example[]>("/api/examples").then(setExamples).catch((e) => setError(String(e))); }, []);

  async function load(id: string) {
    setResult(null); setStatus(null); setRunId(null); setBrief(""); setError("");
    const m = await api<Model>(`/api/examples/${id}`);
    setModel(m);
    setReport(await api<Report>("/api/validate", { model: m }));
  }

  async function build() {
    setError("");
    try {
      const r = await api<{ model: Model; report: Report }>("/api/build", { question });
      setModel(r.model); setReport(r.report); setResult(null);
    } catch (e) { setError(String(e)); }
  }

  async function run() {
    if (!model) return;
    setResult(null); setBrief("");
    const { id } = await api<{ id: number }>("/api/runs", {
      model, scenario: model.scenarios[0]?.name ?? null, workers: 4,
    });
    setRunId(id);
  }

  useEffect(() => {
    if (runId === null) return;
    const timer = setInterval(async () => {
      const s = await api<Status>(`/api/runs/${runId}`);
      setStatus(s);
      if (s.status === "converged" || s.status === "max_batches") {
        clearInterval(timer);
        setResult(await api<Result>(`/api/runs/${runId}/result`));
      }
    }, 300);
    return () => clearInterval(timer);
  }, [runId]);

  const out = model?.outputs[0] ?? "";
  const unit = model ? [...model.aux, ...model.stocks].find((x) => x.name === out)?.unit ?? "" : "";

  return (
    <main>
      <header>
        <h1>whatif-sim</h1>
        <p className="muted">Question → validated stock-and-flow model → Monte Carlo with provenance</p>
      </header>

      <section className="card">
        <label htmlFor="q">Ask a what-if question (needs an API key on the server)</label>
        <div className="row">
          <input id="q" value={question} onChange={(e) => setQuestion(e.target.value)}
            placeholder="What if the job-finding rate falls 20%?" />
          <button onClick={build} disabled={!question}>Build model</button>
        </div>
        <div className="row">
          <span className="muted">or open an example:</span>
          {examples.map((e) => <button key={e.id} className="ghost" onClick={() => load(e.id)}>{e.name}</button>)}
        </div>
        {error && <pre className="error">{error}</pre>}
      </section>

      {model && report && (
        <section className="card">
          <h2>{model.name}</h2>
          <p>{model.question}</p>
          <div className="grid2">
            <LoopDiagram report={report} model={model} />
            <div>
              <h3>Validator</h3>
              <p className={report.ok ? "ok" : "error"}>{report.ok ? "passed" : "rejected"}</p>
              <ul className="issues">
                {report.issues.map((i, k) => <li key={k} className={i.severity}><code>{i.code}</code> {i.where}: {i.message}</li>)}
              </ul>
              <h3>Feedback loops</h3>
              <ul className="loops">
                {report.loops.map((l, k) => (
                  <li key={k}><b className={l.polarity}>{l.polarity}</b> {l.nodes.join(" → ")}</li>
                ))}
              </ul>
              <button onClick={run} disabled={!report.ok || (status !== null && !result)}>
                Run Monte Carlo{model.scenarios[0] ? `: ${model.scenarios[0].name}` : ""}
              </button>
            </div>
          </div>
        </section>
      )}

      {status && (
        <section className="card">
          <h3>Run {runId}: {status.status}</h3>
          <p className="muted">
            {status.prefix_done} batches checkpointed
            {status.convergence && <> · effect {(100 * status.convergence.mean.est).toFixed(3)} pp ±{" "}
              {(100 * status.convergence.mean.halfwidth).toFixed(4)} · p95 half-width{" "}
              {(100 * status.convergence.p95.halfwidth).toFixed(4)} pp</>}
            {status.retried_claims > 0 && <> · {status.retried_claims} batches recomputed after worker loss</>}
          </p>
        </section>
      )}

      {result && model && (
        <section className="card">
          <p className="muted">{result.samples.toLocaleString()} parameter draws in {result.batches} batches</p>
          <div className="grid2">
            <FanChart t={result.t} unit={unit} title={`${out}: baseline vs scenario`} series={[
              { label: "baseline", fan: result.baseline[out], color: "var(--base)" },
              ...(result.scenario ? [{ label: "scenario", fan: result.scenario[out], color: "var(--scen)" }] : []),
            ]} />
            {result.effect && (
              <FanChart t={result.t} unit={unit} title={`${out}: effect (scenario − baseline, paired)`}
                series={[{ label: "effect", fan: result.effect[out], color: "var(--hot)" }]} />
            )}
          </div>
          <button onClick={async () => setBrief(await api<string>("/api/brief", { run_id: runId }))}>Write brief</button>
          {brief && <pre className="brief">{brief}</pre>}
        </section>
      )}
    </main>
  );
}
