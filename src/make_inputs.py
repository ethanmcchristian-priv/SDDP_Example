"""
make_inputs.py  --  Generate the canonical Phase 1 input file for Small SDDP.

This produces `data/inputs_v1.json`: a 10-stage, 10-scenario hydro-thermal system
with 5 hydro plants, 1 thermal plant, and demand. ALL units are MWh for energy /
storage / inflow (per stage), and $/MWh for costs.

The data is deliberately SYNTHETIC and DETERMINISTIC -- there is no random number
generator. Every value comes from a small, readable formula built out of:

  * a per-plant base inflow (the plant's "personality"),
  * a stage seasonality shape (wet early -> dry mid-horizon),
  * a per-scenario hydrology multiplier (scenario 0 = driest, 9 = wettest),
  * a per-scenario/stage thermal cost (dry scenarios -> expensive fuel).

Because it is a pure formula, the file is fully reproducible and you can reason
about every number by hand. Edit the dials below to change the example.

Run:  python src/make_inputs.py
"""

import json
import os

STAGES = 10
SCENARIOS = 10

# ---------------------------------------------------------------------------
# Demand (MWh per stage).  Peaks mid-horizon -- on purpose -- so that demand is
# highest exactly when seasonal inflow is lowest.  That tension is what gives
# stored water a positive marginal value.
# ---------------------------------------------------------------------------
LOAD = [100, 105, 115, 125, 135, 140, 135, 120, 110, 105]

# Stage seasonality applied to inflows: high early (snowmelt/spring), dip mid
# (dry summer), partial recovery late.  Multiplies every plant's base inflow.
SEASONALITY = [1.30, 1.25, 1.15, 1.00, 0.85, 0.70, 0.65, 0.75, 0.90, 1.05]

# Mild stage cost shape for thermal fuel: fuel a bit pricier at peak-demand
# stages (mirrors the load shape).  Multiplies the thermal cost.
STAGE_COST_FACTOR = [0.95, 0.97, 1.00, 1.03, 1.07, 1.10, 1.07, 1.00, 0.97, 0.95]

VOLL = 1000          # $/MWh, value of lost load (unserved energy penalty)
THERMAL_CAPACITY = 120   # MWh per stage; sized so thermal (not blackouts) is the
                         # marginal backstop, so water value tracks fuel cost
THERMAL_COST_REF = 50    # $/MWh reference (actual cost varies by scenario/stage)


def linspace(a, b, n):
    """Evenly spaced list of n values from a to b inclusive."""
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


# Per-scenario hydrology multiplier: scenario 0 driest (0.55x) ... 9 wettest (1.45x).
HYDRO_MULT = linspace(0.55, 1.45, SCENARIOS)

# Per-scenario thermal cost multiplier, INVERSELY correlated with hydrology:
# dry scenarios (0) -> expensive fuel (1.40x), wet scenarios (9) -> cheap (0.70x).
COST_MULT = linspace(1.40, 0.70, SCENARIOS)

# ---------------------------------------------------------------------------
# Exogenous Markov chain over the 10 scenario-states, used ONLY by the
# stagewise-DEPENDENT model. We do NOT estimate this from data -- we assume it
# was found exogenously (simplification). It encodes hydrological PERSISTENCE:
# a dry state tends to be followed by another dry state, so droughts cluster.
# The stagewise-INDEPENDENT model ignores this and redraws from the marginal
# scenario probabilities every stage.
# ---------------------------------------------------------------------------
PERSISTENCE_RHO = 0.6      # 0 = memoryless (uniform-ish), ->1 = very sticky
INITIAL_STATE_CENTER = 5   # start near "near-normal-wet"


def _persistence_row(center, n, rho):
    """Unnormalized weights peaked at `center`, decaying like rho**distance."""
    w = [rho ** abs(j - center) for j in range(n)]
    tot = sum(w)
    return [round(x / tot, 4) for x in w]


def _renorm(row):
    """Fix tiny rounding so the row sums to exactly 1.0 (adjust the max entry)."""
    diff = round(1.0 - sum(row), 4)
    k = row.index(max(row))
    row = list(row)
    row[k] = round(row[k] + diff, 4)
    return row


def build_markov():
    transition = [_renorm(_persistence_row(i, SCENARIOS, PERSISTENCE_RHO))
                  for i in range(SCENARIOS)]
    initial = _renorm(_persistence_row(INITIAL_STATE_CENTER, SCENARIOS, PERSISTENCE_RHO))
    return transition, initial

# ---------------------------------------------------------------------------
# The five hydro plants, each with a distinct "personality" so their water
# values come out meaningfully different.
#   base      : mean inflow (MWh/stage) before seasonality & scenario scaling
#   var_scale : extra spread of the scenario multiplier around 1.0
#               (1.0 = use HYDRO_MULT as-is; >1 amplifies dry/wet swings)
# ---------------------------------------------------------------------------
# Plant sizing targets:
#   * Normal scenario: inflow ~= max_turbine so storage is roughly stable.
#   * Dry scenario (mult 0.55): reservoir depletes toward min_storage but does
#     NOT hit zero -- operators always keep an environmental/safety buffer.
#   * Wet scenario (mult 1.45): inflow exceeds turbine so reservoir fills; spill
#     opens once it hits max_storage.
#   * min_storage > 0 encodes the operational buffer (environmental flows, flood
#     control, instrument range) -- a real constraint that also prevents the model
#     from corner solutions at zero.
#
# Columns: name, role, max_stor, init_stor, min_stor, max_turb, base, var_scale
PLANTS = [
    ("H1", "big reservoir, low inflow (storage-dominated, high water value)",
     200, 90, 30, 18, 16, 1.0),
    ("H2", "medium reservoir, medium inflow (balanced)",
     120, 50, 15, 18, 22, 1.0),
    ("H3", "small reservoir, high inflow (run-of-river, spills, low water value)",
     40,  15,  5, 20, 28, 1.0),
    ("H4", "medium reservoir, highly variable inflow (most scenario-sensitive)",
     150, 60, 20, 18, 22, 2.0),
    ("H5", "small reservoir, low inflow (constrained peaker)",
     60,  25,  8, 10, 12, 1.0),
]


def scenario_mult(var_scale, s):
    """Hydrology multiplier for a plant, with its variance amplified around 1.0."""
    return 1.0 + var_scale * (HYDRO_MULT[s] - 1.0)


def build():
    hydro = []
    inflows = {}
    for name, role, max_stor, init_stor, min_stor, max_turb, base, var_scale in PLANTS:
        hydro.append({
            "name": name,
            "role": role,
            "max_storage_MWh": max_stor,
            "initial_storage_MWh": init_stor,
            "min_storage_MWh": min_stor,
            "max_turbine_MWh_per_stage": max_turb,
            "production_factor_MWh_per_MWh": 1.0,  # inflow already in energy terms
            "downstream": None,                    # no cascade in v1 (field reserved)
        })
        # inflows[name] is a [stage][scenario] grid.
        grid = []
        for t in range(STAGES):
            row = []
            for s in range(SCENARIOS):
                val = base * SEASONALITY[t] * scenario_mult(var_scale, s)
                row.append(round(val))
            grid.append(row)
        inflows[name] = grid

    # Thermal cost: [stage][scenario] grid, $/MWh.
    thermal_cost = []
    for t in range(STAGES):
        row = []
        for s in range(SCENARIOS):
            row.append(round(THERMAL_COST_REF * COST_MULT[s] * STAGE_COST_FACTOR[t]))
        thermal_cost.append(row)

    data = {
        "meta": {
            "name": "small-sddp-v1",
            "description": (
                "Hand-crafted synthetic SDDP example. The goal is to show that a "
                "reservoir's water value is an assumption-dependent output, not a "
                "physical constant. Physical system is fixed; variants change the "
                "expectations/preferences/structure (see PLAN.md, experiments/)."
            ),
            "units": ("Energy, storage, inflow and capacity are MWh per stage. "
                      "Costs are $/MWh. Stored water is expressed directly in MWh "
                      "(production_factor = 1.0)."),
            "stages": STAGES,
            "scenarios": SCENARIOS,
            "stage_label": "week",
            "discount_rate_per_stage": 0.0,   # a variant lever (see V-DISCOUNT)
        },
        "demand": {
            "unserved_energy_cost_per_MWh": VOLL,
            "load_MWh_per_stage": LOAD,
        },
        "thermal": [
            {
                "name": "T1",
                "capacity_MWh_per_stage": THERMAL_CAPACITY,
                "cost_reference_per_MWh": THERMAL_COST_REF,
                "note": "actual variable cost is stochastic; see uncertainty.thermal_cost_per_MWh",
            }
        ],
        "hydro": hydro,
        "uncertainty": {
            "type": "stagewise_independent",
            "indexing": ("All arrays are [stage][scenario] (outer index = stage 0..9, "
                         "inner = scenario 0..9). Read DOWN a column to see one "
                         "scenario's profile over time; scenario 0 = driest / costliest "
                         "fuel, scenario 9 = wettest / cheapest fuel."),
            "scenario_probabilities": [round(1.0 / SCENARIOS, 4)] * SCENARIOS,
            "scenario_labels": [
                "driest", "very dry", "dry", "below-normal", "near-normal-dry",
                "near-normal-wet", "above-normal", "wet", "very wet", "wettest",
            ],
            "inflows_MWh": inflows,
            "thermal_cost_per_MWh": {"T1": thermal_cost},
        },
    }

    transition, initial = build_markov()
    data["uncertainty"]["markov"] = {
        "note": ("Exogenous (assumed, NOT estimated) Markov chain over the 10 "
                 "scenario-states. Used by the stagewise-DEPENDENT model only. "
                 "Stage-independent transition matrix; encodes hydrological "
                 "persistence so droughts cluster. transition_matrix[i][j] = "
                 "P(next state = j | current state = i)."),
        "persistence_rho": PERSISTENCE_RHO,
        "initial_distribution": initial,
        "transition_matrix": transition,
    }
    return data


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, os.pardir, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.normpath(os.path.join(out_dir, "inputs_v1.json"))
    data = build()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
