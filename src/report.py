"""
report.py  --  Train both SDDP models, then simulate fixed-scenario paths and
return storage + water-value trajectories as pandas DataFrames.

A "fixed-scenario path" holds the same scenario index at every stage:
  - scenario 0 = driest (worst inflow, most expensive thermal)
  - scenario 9 = wettest (best inflow, cheapest thermal)

This lets us answer: given a trained policy, how does the system behave
-- and what do water values look like -- in a consistently dry vs consistently
wet realisation?

Usage as a module:
    from report import build_report_data
    df_wv, df_stor = build_report_data()

Usage standalone (outputs JSON for external plotting):
    python src/report.py
"""

import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import data_io
import sddp as sddp_mod


SCENARIO_LABELS = {0: "low inflow (dry)", 9: "high inflow (wet)"}


def simulate_fixed(model, scenario_idx):
    """Run the trained policy with the SAME scenario at every stage.

    Returns:
        stor  : list of dicts {plant: storage_MWh} for stages 0..T  (T+1 entries
                -- includes initial storage before stage 0 as stage -1 for plotting)
        wv    : list of dicts {plant: water_value} for stages 0..T-1
    """
    x = dict(model.x0)
    stor = [dict(x)]           # storage entering stage 0 (initial)
    wv = []

    for t in range(model.T):
        _, _, s_end, fix_dual = model.subproblem(t, scenario_idx, x)
        wv.append({h: -fix_dual[h] for h in model.plants})
        stor.append(s_end)    # storage leaving stage t (entering t+1)
        x = s_end

    return stor, wv


def build_report_data(seed=12345):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, os.pardir, "data", "inputs_v1.json"))
    data = data_io.validate(data_io.load(path))
    plants = [h["name"] for h in data["hydro"]]
    T = data["meta"]["stages"]

    # Three models on the SAME physical system:
    #   independent : believes all 10 scenarios, no memory
    #   markov      : believes all 10, with drought persistence
    #   dry_only    : pessimist -- believes ONLY the 5 driest scenarios (0..4),
    #                 uncorrelated. Same physics, a deliberately truncated future.
    DRY_SCENARIOS = [0, 1, 2, 3, 4]
    END_VALUE = (1000, 35, 0)   # marginal $/MWh at (min, initial, max) storage
    model_specs = [
        ("independent", dict(mode="independent", end_value_points=END_VALUE)),
        ("markov", dict(mode="markov", end_value_points=END_VALUE)),
        ("dry_only", dict(mode="independent", active_scenarios=DRY_SCENARIOS,
                          end_value_points=END_VALUE)),
    ]
    models = {}
    for name, kwargs in model_specs:
        random.seed(seed)
        m = sddp_mod.SDDP(data, **kwargs)
        m.train(iters=40, verbose=False)
        models[name] = m

    max_stor = {h["name"]: h["max_storage_MWh"] for h in data["hydro"]}
    min_stor = {h["name"]: h["min_storage_MWh"] for h in data["hydro"]}
    total_max = sum(max_stor.values())
    total_min = sum(min_stor.values())
    total_init = sum(model.x0.values() for model in models.values().__iter__().__next__().__class__
                     ) if False else sum(data["hydro"][i]["initial_storage_MWh"] for i in range(len(plants)))

    records = []   # one row per (scenario, model, stage) with aggregated metrics

    for scn_idx, scn_label in SCENARIO_LABELS.items():
        for mode, model in models.items():
            stor, wv = simulate_fixed(model, scn_idx)

            for t in range(T):
                avg_wv = sum(wv[t][p] for p in plants) / len(plants)
                total_stor_end = sum(stor[t + 1][p] for p in plants)
                records.append({
                    "scenario": scn_label,
                    "model": mode,
                    "stage": t,
                    "avg_water_value": round(avg_wv, 2),
                    "total_storage_MWh": round(total_stor_end, 1),
                    "total_storage_pct": round(100 * total_stor_end / total_max, 1),
                    # per-plant storage for the storage breakdown chart
                    **{f"stor_{p}": round(stor[t + 1][p], 1) for p in plants},
                    **{f"wv_{p}": round(wv[t][p], 2) for p in plants},
                })

    df = pd.DataFrame(records)
    return df, {
        "total_max_MWh": total_max,
        "total_min_MWh": total_min,
        "total_init_MWh": total_init,
        "plants": plants,
        "max_stor": max_stor,
        "min_stor": min_stor,
    }


if __name__ == "__main__":
    df, meta = build_report_data()
    out = {
        "rows": df.to_dict(orient="records"),
        "meta": meta,
    }
    print(json.dumps(out))
