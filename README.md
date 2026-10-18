# whatif-sim

Ask a "what if" question in plain language. An LLM drafts a small
stock-and-flow model in three stages, **deterministic validators decide**
whether each stage is acceptable (units, structure, a test simulation), and
code, not the model, fills in every number that claims to be data. The model
then runs as a Monte Carlo over parameter uncertainty with an adaptive ODE
solver. Runs checkpoint to Postgres and resume after a crash, and the output
is a brief in which every input says where it came from.

```
question ──► LLM stage 1: structure ──► validator ──┐  rejected? error codes go back
             LLM stage 2: equations ──► validator ──┤  as an is_error tool_result;
             LLM stage 3: evidence map ─► validator ─┘  up to 2 repairs per stage
                                │
                     code computes parameters from the BLS/FRED snapshot
                                │
             Monte Carlo (RK45, common random numbers, batch-means stopping)
             workers ◄──► Postgres (claim / checkpoint / fence / resume)
                                │
             Sobol sensitivity ─► brief (Markdown, provenance per input)
                                │
             FastAPI ◄──► React (D3 causal-loop diagram, fan charts)
```

Python 3.12 (numpy, pydantic, psycopg, FastAPI), Postgres 16, React 19 + D3,
Anthropic Messages API (tool use for structured output, prompt caching).

## The worked example

*"What happens to US unemployment over the next 24 months if the rate at which
unemployed people find jobs falls 20% from month 3?"*
([examples/labor_market.json](examples/labor_market.json))

There are two stocks, employed and unemployed, and two flows between them,
hires and separations. A demand feedback adds a third link: higher
unemployment means less spending, and so fewer hires. The validator finds
6 feedback loops and signs each link from the equations. The demand loop comes
out **reinforcing**; the stock-flow drains come out **balancing**.

The job-finding and separation rates come from 36 months of BLS Current
Population Survey labor-force flows. They use the same series ids FRED
publishes: LNS17100000 / LNS13000000 and LNS17400000 / LNS12000000, for
2023-08 to 2026-09. Monthly rates are computed as flow / previous month's
level, and a lognormal is fitted to them. The one parameter with no data,
`demand_sensitivity`, is a labelled assumption with a stated range.

**Answer** ([results/brief_labor_market.md](results/brief_labor_market.md)):
by month 24, unemployment is **+0.95 pp** above baseline. In 90% of runs the
effect is between +0.78 and +1.17 pp. That comes from 5,376 draws, which the
stopping rule chose. Sobol analysis says the separation rate drives most of
the uncertainty (total index 0.53), ahead of the job-finding rate (0.30) and
the demand assumption (0.23).

## Results

All numbers come from `tools/bench.py` on a laptop (i5-9300H, 4 cores /
8 threads, Postgres on the same machine). Every row has a file in
[results/](results/).

| What | Result |
|---|---|
| **ODE solver.** Dormand–Prince 5(4), adaptive, vectorized over a whole batch | Observed order on 3 problems with exact solutions: **5.05–5.16** at the finest steps (expected 5). Error tracks the tolerance: decay error at rtol 1e-3 → 1e-11 is 3.3e-6 → 1.5e-14. The scenario switch-on lands exactly on t=3, matching the piecewise closed form to 1e-9. |
| **Common random numbers.** Baseline and scenario run on the same draws | The effect's variance is **27× lower** than with independent draws (1.46e-6 vs 3.91e-5). So the same confidence interval needs 27× fewer samples. |
| **Convergence-based stopping.** Batch-means CI on the effect's mean, p5 and p95 | Tolerance 1e-4 needs 21 batches (5,376 draws); 5e-5 needs 85 (21,760). Halving the tolerance costs 4× the samples, as √n predicts. The batch-means half-width matches the i.i.d. one (1.59e-5 vs 1.60e-5). |
| **Sobol indices.** Saltelli/Jansen estimators with bootstrap CIs | Ishigami function (exact indices known): max error 0.028 at N=1k, **0.003 at N=100k**. Labor model (N=4096: 20,480 parameter sets, each run as baseline + scenario, in 1.1 s): ST = sep_rate 0.53 [0.51, 0.55], find_rate 0.30 [0.28, 0.31], demand_sensitivity 0.23 [0.22, 0.25]. |
| **Validator.** 24 seeded model bugs, the kind an LLM plausibly writes | **24/24 rejected with the expected error code.** They cover units per year vs per month, people vs kpeople, a bare number added to a flow, `__import__`, attribute access, algebraic loops, a stock that goes negative, runaway feedback that overflows, and a data source with no series. |
| **Crash recovery.** 4 workers, 13–20 random `TerminateProcess` kills per run, 5 runs | Output was **bit-identical** to an unkilled single-worker run in **10/10** runs (SHA-256 over every sample). |
| **Dead-claim detection.** Lease expiry vs "lease expired or the claimant's Postgres session is gone" | Batches computed past the stop point: **77–103 per run → 1**. Two lease-only runs took 125 s instead of 5 s, because a worker was killed while holding the summary claim. |
| **Parallel workers.** 384 batches, fixed work | Batch phase speeds up 1.7× / 2.4× / 2.8× at 2 / 4 / 6 workers, and end-to-end wall time 1.48× / 1.74× / 1.86×. Per-batch compute rises from 18 to 44 ms as 4 physical cores are shared with Postgres. The rest is the serial summary step plus process start-up. |
| **Tests** | 88 pytest tests (the Postgres ones run in CI against a service container) and 5 vitest tests. |

## Five things that went wrong and were fixed

1. **Sobol gave a negative first-order index (−0.19)** for the job-finding
   rate. The estimators are shift-invariant in expectation, but the effect's
   mean is about 10× its standard deviation, which inflates the estimator
   variance by mean²/var. Centring the outputs fixed it: −0.19 → 0.18 on the
   same draws.
2. **Killed workers stalled the stopping rule.** A dead worker's batch stayed
   "claimed" for its whole lease. The finished prefix couldn't advance, so the
   stopping rule couldn't fire, while the live workers computed batches nobody
   would use. Fix: a claim also expires when its `backend_pid` disappears from
   `pg_stat_activity`. A killed process drops its connection immediately.
3. **Every idle worker built the final summary.** Each one loaded about
   115 MB of trajectories and computed percentiles, the results were thrown
   away, and the copies fought each other for CPU, so the summary took 6 s
   instead of 3. The summary is now claimed like a batch: one winner, and
   takeover if it dies. 4-worker wall time went from 11.4 s to 7.9 s.
4. **Stop checks were O(n²) per batch.** Each check re-tested every prefix
   length. Stats are immutable once written, so it now resumes from
   `checked_upto`.
5. **A gap in the data.** BLS has no October 2025 CPS (shown as "-"). Pairing
   adjacent rows would divide November's flow by September's level. Rates now
   look up the previous *calendar* month, and a test covers the gap. On this
   snapshot the bug was latent: BLS also withholds the November flows.

## Design notes

- **The LLM never supplies a number that claims to be data.** In stage 3 it
  only picks series ids from an allowlist and one of four named transforms.
  The transform's output unit is checked against the parameter's unit before
  anything is computed, then `evidence.fill()` computes the value from the
  snapshot. A number the model puts in a data-backed parameter is discarded,
  and a test checks that.
- **Equations are never `eval`'d as written.** They are parsed with `ast`,
  every node is checked against a whitelist, units are inferred over the tree,
  and only then is the tree compiled with an empty `__builtins__`.
- **Batch i is a pure function of (model, config, i)**, via
  `SeedSequence(seed, spawn_key=(i,))`. Any worker can compute any batch, a
  recomputed batch is bit-identical, and the stopping rule uses the *smallest*
  converged prefix, so the answer doesn't depend on scheduling.
- **Fencing.** `attempt` is a fencing token. A worker that stalls past its
  lease and wakes up can't overwrite a batch someone else finished; the late
  write is recorded in `fenced`.
- **Prompt caching.** The API caches the prefix in the order tools → system →
  messages, so every call sends all three stage tools, identical, and picks
  one with `tool_choice`. The cache breakpoint is on the system prompt. A test
  asserts that the tools and system prompt are byte-identical across all calls
  in a build, including repairs.

## Honest limits

- **The live LLM path has not been run.** This machine has no API key. The
  pipeline is tested end to end with `ScriptedProvider`, which replays
  **hand-written** tool inputs, not recorded model output. The script makes
  two mistakes: a rate per year in a monthly model, then an unknown series plus
  an assumption with no justification. That drives the repair loop through
  both. `python -m whatif build "..."` with `ANTHROPIC_API_KEY` set runs the
  real `AnthropicProvider`. Cache-hit rates have not been measured.
- **The baseline drifts.** The model has no flows into or out of the labor
  force, so its steady state s/(s+f) = 3.6% sits below today's 4.2%, and the
  baseline falls to a 3.56% median by month 24. The what-if *effect* is the
  quantity to read; it is paired and largely cancels the drift.
- **Uncertainty is month-to-month variability.** Parameter distributions are
  the spread of 36 monthly rates, which is a proxy for which regime the next
  two years look like. They are not posterior uncertainty in a mean.
- **Data source.** fred.stlouisfed.org timed out from this machine and from
  GitHub's runners, so the snapshot came from the BLS API under the same
  series ids. With a `FRED_API_KEY` the fetcher uses the FRED API instead.
  The manifest records which source served each file.
- **No screenshot.** The UI was smoke-tested through its API with the built
  bundle served by FastAPI, not checked visually in a browser.

## Running it

```bash
pip install -e ".[dev,llm]"
python -m whatif validate examples/labor_market.json        # issues + loops
python -m whatif local examples/labor_market.json --scenario find_rate_down_20
python -m whatif sobol examples/labor_market.json --scenario find_rate_down_20
python -m whatif brief examples/labor_market.json --scenario find_rate_down_20

# Postgres-checkpointed runs (WHATIF_DB, default postgresql://whatif:whatif@localhost:5433/whatif)
python -m whatif run examples/labor_market.json --scenario find_rate_down_20 --workers 4
python -m whatif chaos examples/labor_market.json --scenario find_rate_down_20 --kills 20

# LLM builder: scripted (no key) or live
python -m whatif build "What if job finding falls 20%?" --script tests/fixtures/scripted_labor.json
ANTHROPIC_API_KEY=... python -m whatif build "What if mortgage rates rise 2 points?"

# Data snapshot and the app
python -m whatif fred-fetch && python -m whatif fred-fill examples/labor_market.template.json examples/labor_market.json
(cd web && npm ci && npm run build) && python -m whatif serve     # http://127.0.0.1:8000

# Tests and benchmarks
WHATIF_TEST_DB=postgresql://... pytest -q
python tools/bench.py solver|order|crn|sobol|stopping|validator|workers|chaos
```

Layout: `whatif/` holds `units`, `expr` (whitelisted expressions + unit
inference), `model` (spec), `validate`, `solver` (RK45), `sim`, `mc`,
`sobol`, `store` (Postgres runs), `evidence` (data + provenance), `brief`,
`llm/` (prompts, providers, pipeline), `api` and `cli`. `web/` is the React
app. `tests/mutations.py` holds the seeded validator bugs.
