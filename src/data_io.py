"""
data_io.py  --  Load, validate, and summarize a Small SDDP input file.

Validation is the acceptance test for Phase 1: if `inputs_v1.json` loads, passes
every check, and the sanity report looks reasonable, the input file is "done".

Run:  python src/data_io.py [path/to/inputs.json]
      (defaults to data/inputs_v1.json)
"""

import json
import os
import sys


class ValidationError(Exception):
    pass


def _check(cond, msg):
    if not cond:
        raise ValidationError(msg)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate(d):
    """Raise ValidationError on the first problem found; return d on success."""
    meta = d["meta"]
    T = meta["stages"]
    S = meta["scenarios"]
    _check(T > 0 and S > 0, "stages and scenarios must be positive")

    # Demand
    load_ = d["demand"]["load_MWh_per_stage"]
    _check(len(load_) == T, f"load has {len(load_)} entries, expected {T}")
    _check(all(x >= 0 for x in load_), "negative demand")
    _check(d["demand"]["unserved_energy_cost_per_MWh"] > 0, "VOLL must be positive")

    # Hydro plants
    names = [h["name"] for h in d["hydro"]]
    _check(len(names) == len(set(names)), "duplicate hydro plant names")
    for h in d["hydro"]:
        lo, init, hi = h["min_storage_MWh"], h["initial_storage_MWh"], h["max_storage_MWh"]
        _check(lo <= init <= hi, f"{h['name']}: need min<=initial<=max storage")
        _check(h["max_turbine_MWh_per_stage"] > 0, f"{h['name']}: max_turbine must be > 0")
        _check(h["production_factor_MWh_per_MWh"] > 0, f"{h['name']}: production_factor must be > 0")
        ds = h.get("downstream")
        _check(ds is None or ds in names, f"{h['name']}: downstream '{ds}' is not a known plant")

    # Thermal
    _check(len(d["thermal"]) >= 1, "need at least one thermal plant")
    for t in d["thermal"]:
        _check(t["capacity_MWh_per_stage"] > 0, f"{t['name']}: capacity must be > 0")

    # Uncertainty
    u = d["uncertainty"]
    probs = u["scenario_probabilities"]
    _check(len(probs) == S, f"{len(probs)} scenario probabilities, expected {S}")
    _check(abs(sum(probs) - 1.0) < 1e-6, f"scenario probabilities sum to {sum(probs)}, expected 1")
    _check(all(p >= 0 for p in probs), "negative scenario probability")

    # Inflow grids: one [stage][scenario] grid per hydro plant.
    for name in names:
        _check(name in u["inflows_MWh"], f"missing inflows for {name}")
        grid = u["inflows_MWh"][name]
        _check(len(grid) == T, f"{name}: inflow has {len(grid)} stages, expected {T}")
        for t, row in enumerate(grid):
            _check(len(row) == S, f"{name} stage {t}: {len(row)} scenarios, expected {S}")
            _check(all(x >= 0 for x in row), f"{name} stage {t}: negative inflow")

    # Thermal cost grids: one [stage][scenario] grid per thermal plant.
    for t in d["thermal"]:
        _check(t["name"] in u["thermal_cost_per_MWh"], f"missing thermal cost for {t['name']}")
        grid = u["thermal_cost_per_MWh"][t["name"]]
        _check(len(grid) == T, f"{t['name']}: cost has {len(grid)} stages, expected {T}")
        for st, row in enumerate(grid):
            _check(len(row) == S, f"{t['name']} stage {st}: {len(row)} scenarios, expected {S}")
            _check(all(c >= 0 for c in row), f"{t['name']} stage {st}: negative cost")

    return d


def _minmax(grid):
    flat = [x for row in grid for x in row]
    return min(flat), max(flat)


def sanity_report(d):
    meta = d["meta"]
    T, S = meta["stages"], meta["scenarios"]
    load_ = d["demand"]["load_MWh_per_stage"]
    u = d["uncertainty"]

    lines = []
    lines.append("=" * 70)
    lines.append(f"  SANITY REPORT  --  {meta['name']}")
    lines.append("=" * 70)
    lines.append(f"  {T} stages ({meta['stage_label']}), {S} scenarios, "
                 f"discount/stage = {meta['discount_rate_per_stage']}")
    lines.append(f"  Units: {meta['units']}")
    lines.append("")

    # Capacity vs demand
    turb_total = sum(h["max_turbine_MWh_per_stage"] for h in d["hydro"])
    therm_cap = sum(t["capacity_MWh_per_stage"] for t in d["thermal"])
    peak = max(load_)
    lines.append("  CAPACITY vs DEMAND (MWh/stage)")
    lines.append(f"    peak demand .............. {peak}")
    lines.append(f"    hydro turbine (all 5) .... {turb_total}")
    lines.append(f"    thermal capacity ......... {therm_cap}")
    lines.append(f"    headroom at peak ......... {turb_total + therm_cap - peak} "
                 f"(>=0 means demand is physically meetable if water is available)")
    lines.append(f"    demand shape ............. {load_}")
    lines.append("")

    # Hydro plants
    lines.append("  HYDRO PLANTS")
    lines.append(f"    {'name':<5}{'maxStor':>8}{'init':>6}{'maxTurb':>8}"
                 f"{'inflow min..max':>18}   role")
    tot_init = 0
    for h in d["hydro"]:
        lo, hi = _minmax(u["inflows_MWh"][h["name"]])
        tot_init += h["initial_storage_MWh"]
        lines.append(f"    {h['name']:<5}{h['max_storage_MWh']:>8}"
                     f"{h['initial_storage_MWh']:>6}{h['max_turbine_MWh_per_stage']:>8}"
                     f"{f'{lo}..{hi}':>18}   {h['role']}")
    tot_stor = sum(h["max_storage_MWh"] for h in d["hydro"])
    lines.append(f"    total initial storage .... {tot_init} MWh "
                 f"(of {tot_stor} MWh max, {100*tot_init/tot_stor:.0f}% full)")
    lines.append("")

    # Thermal
    lines.append("  THERMAL")
    for t in d["thermal"]:
        lo, hi = _minmax(u["thermal_cost_per_MWh"][t["name"]])
        lines.append(f"    {t['name']}: capacity {t['capacity_MWh_per_stage']} MWh/stage, "
                     f"variable cost ranges {lo}..{hi} $/MWh across scenarios/stages")
    lines.append(f"    VOLL (unserved) .......... "
                 f"{d['demand']['unserved_energy_cost_per_MWh']} $/MWh")
    lines.append("")

    # Hydrology spread: total inflow per scenario, summed over stages & plants.
    lines.append("  TOTAL INFLOW BY SCENARIO (MWh, summed over all stages & plants)")
    per_scn = [0.0] * S
    for name in u["inflows_MWh"]:
        for row in u["inflows_MWh"][name]:
            for s, v in enumerate(row):
                per_scn[s] += v
    labels = u.get("scenario_labels", [str(i) for i in range(S)])
    for s in range(S):
        bar = "#" * int(per_scn[s] / 60)
        lines.append(f"    s{s} {labels[s]:<16} {per_scn[s]:>7.0f}  {bar}")
    lines.append("")
    lines.append(f"  driest scenario has {per_scn[0]:.0f} MWh total inflow; "
                 f"wettest has {per_scn[-1]:.0f} MWh "
                 f"({per_scn[-1]/per_scn[0]:.1f}x wetter).")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.normpath(os.path.join(here, os.pardir, "data", "inputs_v1.json"))
    path = sys.argv[1] if len(sys.argv) > 1 else default
    d = load(path)
    try:
        validate(d)
    except ValidationError as e:
        print(f"VALIDATION FAILED: {e}")
        sys.exit(1)
    print(f"OK: {path} is valid.\n")
    print(sanity_report(d))


if __name__ == "__main__":
    main()
