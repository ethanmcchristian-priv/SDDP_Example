# Small SDDP — Project Plan

## 1. Purpose & thesis

Build a deliberately small Stochastic Dual Dynamic Programming (SDDP) model for
hydro-thermal scheduling, then build *several variants* of it. The point is **not**
to compute "the" water value of stored hydro energy — it is to demonstrate that the
**water value is not a mathematical absolute**. It is an output that depends on:

- the **expectations** baked into the model (inflow scenarios, demand forecast),
- the **preferences** of the operator (risk aversion, discount rate),
- the **structural assumptions** (end-of-horizon valuation, thermal cost, spill rules,
  number of scenarios, cut sharing between reservoirs),
- and the **algorithm settings** (iterations, sampling).

By holding the physical system fixed and varying these, we show the same reservoir,
on the same day, can be assigned very different marginal values. That is the whole story.

### What is a "water value" here?
In SDDP the future-cost function `V_t(storage)` is approximated by a set of linear
**Benders cuts**. The **water value** of a reservoir at stage `t` is the *slope* of that
future-cost function with respect to that reservoir's stored volume — i.e. the dual /
shadow price `∂V_t/∂storage`. It answers: "what is one more unit of stored water worth,
in expected future cost saved?" Everything downstream of "expected" is an assumption.

---

## 2. The physical system (held constant across all variants)

Small, hand-made example so results are inspectable by eye.

| Element            | Count | Notes                                                        |
|--------------------|-------|-------------------------------------------------------------|
| Stages (timesteps) | 10    | e.g. 10 weeks                                                |
| Scenarios          | 10    | inflow realizations per stage (stagewise independent first) |
| Hydro plants       | 5     | each with reservoir storage + inflow                         |
| Thermal plant      | 1     | the "backstop" — sets the marginal cost when hydro is scarce |
| Demand             | 1     | a single system load per stage                              |

### Hydro plants (5)
Give them distinct personalities so water values differ meaningfully:
- **H1 – Big reservoir, low inflow** (storage-dominated, high water value)
- **H2 – Medium reservoir, medium inflow** (balanced)
- **H3 – Small reservoir, high inflow** (run-of-river-ish, low water value, spills often)
- **H4 – Medium reservoir, seasonal/variable inflow** (most sensitive to scenarios)
- **H5 – Small reservoir, low inflow** (constrained peaker)

Per-plant parameters:
- `max_storage` (volume units, e.g. Mm³ or MWh of energy)
- `initial_storage`
- `min_storage` (often 0)
- `max_turbine` (max release that produces power, per stage)
- `production_factor` (MWh per unit of water released) — converts water → energy
- `max_spill` (usually unbounded / large)
- optional cascade topology (see below — start with **no cascade**)

### Thermal plant (1)
- `capacity` (MW or MWh/stage)
- `cost` (\$/MWh) — this is the price ceiling that anchors water values
- (later variant: add a 2nd thermal block or a price cap / unserved-energy cost)

### Demand
- One demand value per stage (can be flat or have a simple shape).
- `unserved_energy_cost` (Value of Lost Load, VOLL) — very high \$/MWh penalty so the
  model serves load unless physically impossible.

### Inflows
- A 5 (plants) × 10 (stages) × 10 (scenarios) array of inflow values.
- Start with **stagewise-independent** scenarios (each stage draws 1 of 10 inflows,
  equiprobable). This keeps the first SDDP implementation simple.
- Keep the numbers round and hand-authored so the example is legible.

---

## 3. Repository layout

```
Small SDDP/
├── PLAN.md                  # this file
├── README.md                # short overview, how to run
├── data/
│   └── inputs_v1.(json|yaml)# the canonical input file (Phase 1 deliverable)
├── src/
│   ├── model.py             # core SDDP solver
│   ├── data_io.py           # load/validate input file
│   └── report.py            # extract water values, plot/print results
├── experiments/
│   └── variants.md          # catalog of assumption-variants to run
├── results/                 # generated outputs (gitignored except samples)
└── notebooks/               # optional exploration
```

(Language/solver decided in Phase 2 — likely Python. Candidate stacks:
hand-rolled LP with `PuLP`/`Pyomo`+CBC/HiGHS for transparency, or `SDDP.jl` if we
want a battle-tested engine. Recommendation: **roll our own small LP-based SDDP** so
every assumption is visible and editable — that *is* the point of the project.)

---

## 4. Phase 1 — Input file (do this first)

**Deliverable:** one human-readable, version-controlled input file that fully specifies
a 10-stage, 10-scenario, 5-hydro + 1-thermal system with demand.

### Proposed schema (JSON shown; YAML equivalent fine)

```jsonc
{
  "meta": {
    "name": "small-sddp-v1",
    "stages": 10,
    "scenarios": 10,
    "stage_label": "week",
    "discount_rate": 0.0          // per-stage; 0 = no discounting (a variant lever)
  },
  "demand": {
    "unserved_energy_cost": 1000, // $/MWh (VOLL)
    "load_per_stage": [100, 100, 110, 120, 130, 140, 130, 120, 110, 100]
  },
  "thermal": [
    {
      "name": "T1",
      "capacity": 80,             // MWh per stage
      "cost": 50                  // $/MWh
    }
  ],
  "hydro": [
    {
      "name": "H1",
      "max_storage": 200,
      "initial_storage": 150,
      "min_storage": 0,
      "max_turbine": 40,          // water units/stage
      "production_factor": 1.0,   // MWh per water unit
      "downstream": null          // cascade target, null = none
    }
    // ... H2..H5
  ],
  "inflows": {
    "type": "stagewise_independent",
    "probabilities": [0.1, 0.1, ...],   // length 10, sums to 1
    // values[plant][stage][scenario]
    "values": {
      "H1": [[s1..s10], [s1..s10], ...10 stages... ],
      "H2": [...],
      "H3": [...],
      "H4": [...],
      "H5": [...]
    }
  }
}
```

### Phase 1 tasks
1. Decide file format (JSON vs YAML) and units (energy vs volume; pick one and document).
2. Hand-author the 5 hydro plant parameter sets per the "personalities" above.
3. Hand-author the thermal + demand numbers so the system is *interesting*:
   - load should sometimes exceed cheap hydro+thermal so water value is positive,
   - but not always binding, so water value varies across stages/states.
4. Hand-author the 5×10×10 inflow table (round numbers; encode the personalities —
   e.g. H3 high & stable, H4 high-variance).
5. Write `data_io.py` to **load and validate** (shapes, probabilities sum to 1,
   storage bounds sane). Validation is the acceptance test for Phase 1.

**Acceptance for Phase 1:** input file loads, validates, and a quick "sanity report"
prints the system (totals, capacity vs peak demand, inflow ranges).

---

## 5. Phase 2 — Core SDDP model

Build the baseline solver against `inputs_v1`.

### Per-stage LP (the subproblem)
Decision variables per stage `t`, per scenario:
- `release[h]` (turbined water), `spill[h]`, `gen_hydro[h] = production_factor*release`
- `gen_thermal`, `unserved`
- `storage_end[h]` (state out)

Constraints:
- **Water balance:** `storage_end[h] = storage_start[h] + inflow[h,t,scn] - release[h] - spill[h]`
  (+ upstream releases/spills if cascade)
- **Storage bounds:** `min_storage ≤ storage_end[h] ≤ max_storage`
- **Turbine bound:** `0 ≤ release[h] ≤ max_turbine`
- **Demand balance:** `Σ gen_hydro + gen_thermal + unserved = load[t]`
- **Thermal bound:** `0 ≤ gen_thermal ≤ capacity`
- **Future cost:** `α ≥ cut_intercept + Σ_h cut_slope[h]*storage_end[h]` for each cut
- **Objective:** minimize `thermal.cost*gen_thermal + VOLL*unserved + α`

### SDDP loop
- **Backward pass:** for each stage `t` from `T..1`, for each scenario, solve the LP,
  collect duals on the water-balance constraints → these are the **water values** →
  average across scenarios to form a new **cut** added to stage `t-1`'s future-cost fn.
- **Forward pass:** simulate forward through sampled scenarios using current cuts to get
  trial storage trajectories (the states at which to add cuts).
- **Convergence:** compare lower bound (stage-1 objective) to a forward-simulation
  statistical upper bound; stop when within tolerance or max iterations.

### Water-value extraction (the product of the project)
For a given stage and storage state, report:
- the future-cost function `V_t(storage)` (the cut envelope),
- its slope per reservoir = **water value** (\$/MWh or \$/water-unit),
- how it changes across stages and across the 5 plants.

**Acceptance for Phase 2:** converges on `inputs_v1`; prints/plots water values per
plant per stage; results are explainable by hand (e.g. H3 ≈ 0 because it spills,
H1 high because storage is scarce relative to demand).

---

## 6. Phase 3 — Variants (the actual demonstration)

Keep the physical system fixed; change one assumption at a time and compare water values.
Catalog these in `experiments/variants.md`. Candidate levers, grouped by the thesis:

### Expectations (what the future looks like)
- **V-INFLOW-DRY / WET:** shift the inflow table down/up → water becomes scarce/abundant.
- **V-INFLOW-VAR:** same mean, higher variance → water value rises with uncertainty.
- **V-CORRELATED:** replace stagewise-independent with persistent (AR/Markov) inflows
  → droughts cluster → reservoirs worth more.
- **V-DEMAND:** higher / peakier demand forecast.

### Preferences (operator attitude)
- **V-DISCOUNT:** discount rate 0% vs 5% vs 10% per stage → future water worth less.
- **V-RISK:** risk-neutral (expectation) vs **risk-averse** (CVaR / worst-scenario
  weighting) → risk aversion *raises* water values, especially for the storage plants.
- **V-ENDVALUE:** end-of-horizon reservoir valuation: zero vs a salvage value vs a
  target-level penalty → dramatically changes late-stage water values.

### Structure / model choices
- **V-THERMAL-COST:** thermal at \$30 vs \$50 vs \$80 → re-anchors the price ceiling and
  scales all water values.
- **V-VOLL:** value of lost load 1,000 vs 10,000 → changes value of water in tight states.
- **V-SCENARIOS:** 3 vs 10 vs 50 scenarios → sampling changes the estimated value.
- **V-CASCADE:** turn on a hydraulic cascade (H1→H2→H3) → upstream water inherits
  downstream value.

### Output for each variant
A comparison table / chart: **water value per plant per stage**, baseline vs variant,
with a one-line narrative of *why* it moved. The collection of these is the deliverable
that proves the thesis: "no correct water value — it's assumptions all the way down."

---

## 7. Suggested order of work

1. **Phase 1** — agree schema, hand-author `inputs_v1`, write loader+validator+sanity report.
2. **Phase 2** — implement baseline SDDP, get water values out, sanity-check by hand.
3. **Phase 3** — implement variant switches, run the catalog, build the comparison report.
4. Write `README.md` with the headline finding and how to reproduce.

## 8. Open decisions

Resolved for v1:
- [x] File format: **JSON** (`data/inputs_v1.json`).
- [x] Units: **energy (MWh)** throughout; stored water in MWh, `production_factor = 1.0`.
- [x] Scenarios: **10**, each a coherent state of the world carrying both an inflow
      profile (per plant, per stage) **and** a thermal variable cost (per stage).
- [x] Thermal cost varies by **scenario AND stage**; correlated so dry scenarios have
      expensive fuel (a built-in coupled assumption that moves water values).
- [x] Cascade: field `downstream` present but `null` in v1 (deferred to a variant).

Resolved for Phase 2:
- [x] Language/solver: **Python + PuLP**, solved in-process with **HiGHS** (open source),
      falling back to bundled **CBC**. Install: `pip install pulp highspy`.
- [x] Two uncertainty modes in **one engine**: stagewise-**independent** and
      stagewise-**dependent (Markov)** with an exogenous non-uniform transition matrix.
      Independent is the special case where every transition row equals the marginal
      scenario probabilities, so the same code serves both.

## 9. Status
- **Phase 1 — DONE.** `src/make_inputs.py` generates `data/inputs_v1.json`;
  `src/data_io.py` loads, validates, and prints a sanity report.
- **Phase 2 — DONE.** `src/sddp.py` trains both models and reports water values.
  Both converge (LB ≈ simulated cost, gap < 1%). Headline finding already visible:
  the Markov model assigns **systematically higher** water values than the independent
  model on the *same* physical system, because persistent droughts make storage more
  valuable — concrete proof that the water value reflects the assumed future, not a
  physical constant. Results written to `results/water_values.csv` + `summary.json`.
- **Phase 3 — next.** Turn the one-off variants into a sweep (see §6): inflow
  dry/wet/variance, discount rate, risk aversion (CVaR), end-of-horizon value,
  thermal cost level, scenario count, cascade — each as a comparison table vs baseline.

### How the engine works (one paragraph)
Each stage is an LP that fixes incoming storage with `storage_start[h] == x[h]`; the
dual of that constraint is the **water value** (cost saved per extra MWh stored). The
future-cost function is built from **Benders cuts** added in a backward pass; a forward
pass (several sampled paths per iteration) supplies the trial storage states. For the
Markov model each cut is a transition-probability-weighted combination of the successor
states' subproblem values. Convergence is tracked by an expected first-stage **lower
bound** vs a Monte-Carlo **upper bound**.
