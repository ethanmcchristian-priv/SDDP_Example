# Small SDDP

A deliberately small Stochastic Dual Dynamic Programming (SDDP) model for
hydro-thermal scheduling, built to show that a reservoir's **water value is an
assumption-dependent output, not a physical constant**. The physical system stays
fixed while different expectations, preferences, and structural choices are varied —
demonstrating there is no single "correct" water value.

See [PLAN.md](PLAN.md) for the full plan and the catalog of planned variants.

## System (v1)

10 stages (weeks) × 10 scenarios, 5 hydro plants + 1 thermal plant + demand.
All energy/storage/inflow in **MWh per stage**, costs in **$/MWh**. Each scenario is a
coherent state of the world (scenario 0 = driest with expensive fuel → 9 = wettest with
cheap fuel), carrying both an inflow profile and a thermal variable-cost profile.

## Layout

```
data/inputs_v1.json   canonical input file (generated, version-controlled)
src/make_inputs.py     deterministic generator for inputs_v1.json (edit the dials here)
src/data_io.py         load + validate + sanity report (the Phase 1 acceptance test)
src/sddp.py            the LP-based SDDP engine (independent + Markov modes)
results/               generated water-value tables (gitignored)
PLAN.md                project plan and variant catalog
```

## Install

```bash
pip install pulp highspy   # PuLP modeling + in-process HiGHS solver (open source)
```

## Usage

```bash
python src/make_inputs.py     # (re)generate data/inputs_v1.json
python src/data_io.py         # validate and print the sanity report
python src/sddp.py            # train both SDDP models and print the water-value comparison
```

`sddp.py` trains a **stagewise-independent** model and a **stagewise-dependent (Markov)**
model on the same physical system and prints their water values side by side. The Markov
model uses the exogenous, non-uniform transition matrix in the input file (drought
persistence), which raises water values — the first concrete demonstration that the water
value depends on the assumed future, not just the physics.

The input data is hand-crafted synthetic data produced by a pure formula (no RNG), so
every number is reproducible and reason-about-able. Tweak the dials at the top of
`src/make_inputs.py` (demand shape, seasonality, plant personalities, cost correlation)
to change the example.

## Status

- **Phase 1 — done:** input file + loader/validator/sanity report.
- **Phase 2 — done:** LP-based SDDP engine with stagewise-independent and Markov modes;
  both converge and produce comparable water values that differ by assumption.
- **Phase 3 — next:** sweep the remaining assumption levers (risk aversion, discount,
  end-of-horizon value, inflow shifts, scenario count, cascade) — see PLAN.md §6.
