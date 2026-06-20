"""
revenue.py  --  Compute per-scenario hydro producer revenue under each trained
policy and write a CSV + summary JSON.

REVENUE MODEL.  At each stage the LP exposes the dual of the demand-balance
constraint -- the system marginal cost (= competitive uniform-price spot
price). A hydro producer's stage revenue is `price * MWh dispatched`. Total
revenue per (model, scenario) sums across all 5 producers and all 10 stages.

This is the right number for a price-taking producer in a competitive market.
For self-scheduling, pay-as-bid, or contract-for-difference settings the
revenue accounting would differ qualitatively.

Each scenario is run as a FIXED-scenario path (same scenario index every
stage) under the model's trained policy -- "how much would this producer have
made if reality turned out to be scenario k all the way through?"

Run:  python src/revenue.py
"""

import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import data_io
import sddp as sddp_mod


END_VALUE = (1000, 35, 0)         # marginal $/MWh at (min, init, max) storage
DRY_SCENARIOS = [0, 1, 2, 3, 4]   # the 5 driest, used by the pessimist
SCENARIO_LABELS = ["driest", "very dry", "dry", "below-normal", "near-normal dry",
                   "near-normal wet", "above-normal", "wet", "very wet", "wettest"]


def trained_models(data, seed=12345):
    specs = [
        ("independent", dict(mode="independent", end_value_points=END_VALUE)),
        ("markov",      dict(mode="markov",      end_value_points=END_VALUE)),
        ("dry-only",    dict(mode="independent", active_scenarios=DRY_SCENARIOS,
                              end_value_points=END_VALUE)),
    ]
    models = []
    for label, kwargs in specs:
        random.seed(seed)
        m = sddp_mod.SDDP(data, **kwargs)
        m.train(iters=40, verbose=False)
        models.append((label, m))
    return models


def revenue_one_path(model, scenario_idx):
    """Run a fixed-scenario path and accumulate per-plant revenue and volume.
    Returns (per_plant_rev, per_plant_mwh, volume_weighted_avg_price)."""
    plants = model.plants
    x = dict(model.x0)
    rev = {p: 0.0 for p in plants}
    mwh = {p: 0.0 for p in plants}
    price_x_mwh = 0.0
    total_mwh = 0.0
    for t in range(model.T):
        _, _, s_end, _, dec = model.subproblem(t, scenario_idx, x)
        price = dec["price"]
        for p in plants:
            g = dec["hydro_gen"][p]
            rev[p] += price * g
            mwh[p] += g
            price_x_mwh += price * g
            total_mwh += g
        x = s_end
    avg_price = price_x_mwh / total_mwh if total_mwh > 0 else 0.0
    return rev, mwh, avg_price


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data = data_io.validate(data_io.load(
        os.path.normpath(os.path.join(here, os.pardir, "data", "inputs_v1.json"))))
    out_dir = os.path.normpath(os.path.join(here, os.pardir, "results"))
    os.makedirs(out_dir, exist_ok=True)

    models = trained_models(data)
    plants = models[0][1].plants

    csv_rows = []         # one row per (model, scenario, plant)
    totals = {}           # totals[label][scn] = {total_rev, total_mwh, avg_price}
    for label, model in models:
        totals[label] = {}
        for scn in range(model.S):
            rev, mwh, avg_price = revenue_one_path(model, scn)
            for p in plants:
                csv_rows.append({"model": label, "scenario": scn,
                                 "scenario_label": SCENARIO_LABELS[scn],
                                 "plant": p,
                                 "revenue_usd": round(rev[p], 2),
                                 "generation_MWh": round(mwh[p], 2)})
            totals[label][scn] = {
                "total_revenue_usd": round(sum(rev.values()), 2),
                "total_generation_MWh": round(sum(mwh.values()), 2),
                "vol_weighted_capture_price_per_MWh": round(avg_price, 2),
            }

    csv_path = os.path.join(out_dir, "revenue.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    json_path = os.path.join(out_dir, "revenue.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(totals, f, indent=2)

    # Print a readable summary
    labels = [lab for lab, _ in models]
    print("TOTAL HYDRO REVENUE ($) per scenario, summed across all 5 producers and 10 stages")
    print(f"  {'scenario':<22}" + "".join(f"{lab:>14}" for lab in labels))
    print("  " + "-" * (22 + 14 * len(labels)))
    for scn in range(10):
        row = "  " + f"s{scn} {SCENARIO_LABELS[scn]:<18}"
        for lab in labels:
            row += f"{totals[lab][scn]['total_revenue_usd']:>14,.0f}"
        print(row)
    print(f"\n  wrote {csv_path}")
    print(f"  wrote {json_path}")


if __name__ == "__main__":
    main()
