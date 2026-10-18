# What-if brief: labor_market

**Question.** What happens to US unemployment over the next 24 months if the rate at which unemployed people find jobs falls 20% from month 3?

**Scenario.** Job-finding rate falls 20% from month 3 onward (`find_rate` x0.8 from t=3).

## Answer

At month 24, `unemployment_rate` changes by **+0.95 pp** versus baseline (90% of runs between +0.78 pp and +1.17 pp).

| | median at t=24 | 90% band |
|---|---|---|
| baseline | 3.56% | 2.99% - 4.23% |
| scenario | 4.49% | 3.78% - 5.36% |

Monte Carlo: 5,376 parameter draws in 21 batches (converged); the mean effect's 95% half-width is 0.003 pp. Baseline and scenario share the same draws (common random numbers).

## What drives the uncertainty

Sobol indices of the answer above (share of its variance; total includes interactions).

| parameter | first-order | total | source |
|---|---|---|---|
| `sep_rate` | 0.48 | 0.54 | fred |
| `find_rate` | 0.26 | 0.29 | fred |
| `demand_sensitivity` | 0.20 | 0.24 | assumption |

## Structure

- balancing: demand -> hires -> employed -> separations -> unemployed -> unemployment_rate -> demand
- reinforcing: demand -> hires -> employed -> unemployment_rate -> demand
- reinforcing: demand -> hires -> unemployed -> unemployment_rate -> demand
- balancing: employed -> separations -> employed
- reinforcing: employed -> separations -> unemployed -> hires -> employed
- balancing: hires -> unemployed -> hires

## Inputs and where they came from

| parameter | value | source |
|---|---|---|
| `find_rate` [1/month] | lognormal(median 0.2583, gsd 1.065) | BLS CPS LNS17100000, LNS13000000 (same ids on FRED); flow_over_lagged_level: monthly flow_t / level_(t-1) over the window, lognormal fitted (unit: 1/month); window 2023-08-01..2026-09-01 (36 months); fetched 2026-10-18T12:34:20+00:00 |
| `sep_rate` [1/month] | lognormal(median 0.009677, gsd 1.084) | BLS CPS LNS17400000, LNS12000000 (same ids on FRED); flow_over_lagged_level: monthly flow_t / level_(t-1) over the window, lognormal fitted (unit: 1/month); window 2023-08-01..2026-09-01 (36 months); fetched 2026-10-18T12:34:19+00:00 |
| `employed0` [kpeople] | 163152 | BLS CPS LNS12000000 (same ids on FRED); latest: latest observation of the series, as a fixed value; window 2026-09-01; fetched 2026-10-18T12:34:19+00:00 |
| `unemployed0` [kpeople] | 7109 | BLS CPS LNS13000000 (same ids on FRED); latest: latest observation of the series, as a fixed value; window 2026-09-01; fetched 2026-10-18T12:34:20+00:00 |
| `u_ref` [1] | 0.041754 | BLS CPS LNS13000000, LNS12000000 (same ids on FRED); share_latest: latest a / (a + b), as a fixed value; window 2026-09-01; fetched 2026-10-18T12:34:19+00:00 |
| `demand_sensitivity` [1] | uniform(0, 5) | **assumption**: No direct data. Range means a 1 point rise in unemployment cuts job finding by 0-5%. Sobol shows how much the answer depends on it. |
| `demand_floor` [1] | 0.2 | **assumption**: Guard so hiring never goes to zero; not reached in any scenario here. |

Run fingerprint: `e922d7971eb3cc3b`
