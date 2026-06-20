"""
sddp.py  --  A small, transparent LP-based SDDP engine for Small SDDP.

One engine, two uncertainty modes:

  * "independent" : stagewise-independent. Each stage the realization is redrawn
                    from the marginal scenario probabilities, with no memory.
  * "markov"      : stagewise-dependent. The scenario-state follows the exogenous
                    Markov chain in the input file (non-uniform transition matrix),
                    so droughts persist.

These are the SAME algorithm. The Markov case is the general one; the independent
case is the special case where every transition row equals the marginal scenario
probabilities. Comparing the two on the SAME physical system is the whole point:
the water value you get out depends on which uncertainty assumption you bring in.

WATER VALUE.  Each subproblem fixes incoming storage with a constraint
`storage_start[h] == x[h]`. The dual of that constraint is d(cost)/d(x[h]) -- the
marginal value of one more MWh stored in plant h. We report the water value as the
cost *reduction* per stored MWh, i.e. minus that dual (a non-negative number).

The LP is solved in-process with HiGHS via PuLP (falls back to CBC).

Run:  python src/sddp.py            # trains both models and prints a comparison
"""

import os
import random

import pulp

import data_io


def make_solver():
    """Prefer in-process HiGHS (fast, clean duals); fall back to bundled CBC."""
    try:
        s = pulp.HiGHS(msg=False)
        return s, "HiGHS"
    except Exception:
        return pulp.PULP_CBC_CMD(msg=False), "CBC"


SOLVER, SOLVER_NAME = make_solver()


class Cut:
    """theta >= intercept + sum_h slope[h] * storage_end[h]."""
    __slots__ = ("intercept", "slope")

    def __init__(self, intercept, slope):
        self.intercept = intercept
        self.slope = slope  # dict: plant name -> coefficient


class SDDP:
    def __init__(self, data, mode):
        assert mode in ("independent", "markov")
        self.mode = mode
        u = data["uncertainty"]
        self.T = data["meta"]["stages"]
        self.S = data["meta"]["scenarios"]
        r = data["meta"]["discount_rate_per_stage"]
        self.df = 1.0 / (1.0 + r)            # one-stage discount factor on the future

        # Physical system
        self.plants = [h["name"] for h in data["hydro"]]
        self.hydro = {h["name"]: h for h in data["hydro"]}
        self.thermal = data["thermal"][0]
        self.therm_name = self.thermal["name"]
        self.cap = self.thermal["capacity_MWh_per_stage"]
        self.voll = data["demand"]["unserved_energy_cost_per_MWh"]
        self.load = data["demand"]["load_MWh_per_stage"]
        self.x0 = {h["name"]: h["initial_storage_MWh"] for h in data["hydro"]}

        # Uncertainty tables: realization in (stage t, state m)
        self.inflow = u["inflows_MWh"]                       # [plant][t][m]
        self.tcost = u["thermal_cost_per_MWh"][self.therm_name]  # [t][m]

        # Transition / initial distribution per mode
        if mode == "markov":
            mk = u["markov"]
            self.initial = mk["initial_distribution"]
            self.P = mk["transition_matrix"]
        else:
            p = u["scenario_probabilities"]
            self.initial = list(p)
            self.P = [list(p) for _ in range(self.S)]  # rows all equal -> memoryless

        # Cut pool: cuts[t][m] is a list of Cut. theta at the LAST stage approximates
        # the end-of-horizon value, which is 0 here (a structural assumption, the
        # V-ENDVALUE lever in PLAN.md), so the last stage never gets cuts.
        self.cuts = [[[] for _ in range(self.S)] for _ in range(self.T)]
        self.lb_history = []

    # ---- sampling helpers -------------------------------------------------
    def sample_initial(self):
        return random.choices(range(self.S), weights=self.initial, k=1)[0]

    def sample_next(self, m):
        return random.choices(range(self.S), weights=self.P[m], k=1)[0]

    # ---- the stage subproblem --------------------------------------------
    def subproblem(self, t, m, x):
        """Solve stage t in state m with incoming storage dict x.

        Returns (total_obj, immediate_cost, storage_end, fix_dual) where
        fix_dual[h] = d(total_obj)/d(x[h])  (<= 0; more water -> lower cost).
        """
        prob = pulp.LpProblem(f"sp_t{t}_m{m}", pulp.LpMinimize)

        s0, s1, rel, spl = {}, {}, {}, {}
        for h in self.plants:
            hp = self.hydro[h]
            s0[h] = pulp.LpVariable(f"s0_{h}", lowBound=None)  # incoming (fixed below)
            s1[h] = pulp.LpVariable(f"s1_{h}", lowBound=hp["min_storage_MWh"],
                                    upBound=hp["max_storage_MWh"])
            rel[h] = pulp.LpVariable(f"rel_{h}", lowBound=0,
                                     upBound=hp["max_turbine_MWh_per_stage"])
            spl[h] = pulp.LpVariable(f"spl_{h}", lowBound=0)
        gen = pulp.LpVariable("therm", lowBound=0, upBound=self.cap)
        uns = pulp.LpVariable("unserved", lowBound=0)
        theta = pulp.LpVariable("theta", lowBound=0)  # future cost (>= end value 0)

        # Fixing constraints: their duals are the incoming water values.
        fix = {}
        for h in self.plants:
            c = pulp.LpConstraint(s0[h], sense=pulp.LpConstraintEQ, rhs=x[h],
                                  name=f"fix_{h}")
            prob += c
            fix[h] = c

        # Water balance (no cascade in v1): s1 = s0 + inflow - release - spill
        for h in self.plants:
            inflow = self.inflow[h][t][m]
            prob += (s1[h] == s0[h] + inflow - rel[h] - spl[h]), f"wb_{h}"

        # Demand balance: hydro energy + thermal + unserved == load
        hydro_energy = pulp.lpSum(
            self.hydro[h]["production_factor_MWh_per_MWh"] * rel[h] for h in self.plants)
        prob += (hydro_energy + gen + uns == self.load[t]), "demand"

        # Cuts on the future cost
        for cut in self.cuts[t][m]:
            prob += (theta >= cut.intercept
                     + pulp.lpSum(cut.slope[h] * s1[h] for h in self.plants)), ""

        # Objective: immediate variable cost + discounted future cost
        immediate = self.tcost[t][m] * gen + self.voll * uns
        prob += immediate + self.df * theta

        prob.solve(SOLVER)

        storage_end = {h: s1[h].value() for h in self.plants}
        fix_dual = {h: fix[h].pi for h in self.plants}
        immediate_val = self.tcost[t][m] * gen.value() + self.voll * uns.value()
        return pulp.value(prob.objective), immediate_val, storage_end, fix_dual

    # ---- one training iteration ------------------------------------------
    def forward(self):
        """Single forward simulation. Returns the per-stage storage_end (trial
        points for the backward pass) and the discounted immediate cost."""
        x = dict(self.x0)
        m = self.sample_initial()
        trial = []
        cost = 0.0
        for t in range(self.T):
            obj, immediate, s_end, _ = self.subproblem(t, m, x)
            cost += (self.df ** t) * immediate
            trial.append(s_end)
            x = s_end
            if t < self.T - 1:
                m = self.sample_next(m)
        return trial, cost

    def backward(self, trial):
        """Add cuts. At stage t the cut approximates the expected value of being
        at end-of-stage-t storage y = trial[t], evaluated via the successor
        subproblems at stage t+1. Cuts are added to every node (t, m)."""
        for t in range(self.T - 2, -1, -1):
            y = trial[t]
            # Solve every successor state once at the trial point.
            Q, g = {}, {}
            for n in range(self.S):
                obj_n, _, _, fix_n = self.subproblem(t + 1, n, y)
                Q[n] = obj_n
                g[n] = fix_n  # d(obj_n)/d(incoming storage) = cut gradient
            # Build a P[m][.]-weighted cut for each node (t, m).
            for mm in range(self.S):
                w = self.P[mm]
                slope = {h: sum(w[n] * g[n][h] for n in range(self.S))
                         for h in self.plants}
                intercept = sum(w[n] * (Q[n] - sum(g[n][h] * y[h] for h in self.plants))
                                for n in range(self.S))
                self.cuts[t][mm].append(Cut(intercept, slope))

    def lower_bound(self):
        """Deterministic lower bound: expected first-stage value over the initial
        distribution, using the current cuts."""
        lb = 0.0
        for m in range(self.S):
            if self.initial[m] == 0:
                continue
            obj, _, _, _ = self.subproblem(0, m, self.x0)
            lb += self.initial[m] * obj
        return lb

    def train(self, iters=40, n_paths=5, tol=1e-3, patience=5, verbose=True):
        """Each iteration runs `n_paths` forward simulations (better exploration of
        the low-storage states that actually drive water value) and adds cuts from
        each, then recomputes the lower bound."""
        stable = 0
        for it in range(1, iters + 1):
            for _ in range(n_paths):
                trial, _ = self.forward()
                self.backward(trial)
            lb = self.lower_bound()
            self.lb_history.append(lb)
            if verbose and (it <= 3 or it % 5 == 0):
                print(f"    iter {it:3d}   lower bound = {lb:10.2f}")
            if len(self.lb_history) >= 2:
                prev = self.lb_history[-2]
                if prev > 0 and abs(lb - prev) / abs(prev) < tol:
                    stable += 1
                    if stable >= patience:
                        if verbose:
                            print(f"    converged after {it} iterations "
                                  f"(LB = {lb:.2f}).")
                        break
                else:
                    stable = 0
        return self.lb_history[-1]

    # ---- evaluation -------------------------------------------------------
    def simulate(self, n=200, seed=0):
        """Monte-Carlo simulation with the trained policy. Returns expected cost
        and average water value per (stage, plant). Water value = -fix_dual."""
        rng = random.Random(seed)
        total = 0.0
        wv_sum = [{h: 0.0 for h in self.plants} for _ in range(self.T)]
        count = [0] * self.T
        for _ in range(n):
            x = dict(self.x0)
            m = rng.choices(range(self.S), weights=self.initial, k=1)[0]
            cost = 0.0
            for t in range(self.T):
                _, immediate, s_end, fix_dual = self.subproblem(t, m, x)
                cost += (self.df ** t) * immediate
                for h in self.plants:
                    wv_sum[t][h] += -fix_dual[h]   # cost saved per extra stored MWh
                count[t] += 1
                x = s_end
                if t < self.T - 1:
                    m = rng.choices(range(self.S), weights=self.P[m], k=1)[0]
            total += cost
        exp_cost = total / n
        water_value = [{h: wv_sum[t][h] / count[t] for h in self.plants}
                       for t in range(self.T)]
        return exp_cost, water_value


def _fmt_wv_table(model, water_value):
    lines = [f"  WATER VALUES ($/MWh stored) -- {model.mode} model"]
    header = "    stage  " + "".join(f"{h:>8}" for h in model.plants)
    lines.append(header)
    for t in range(model.T):
        row = f"    {t:>5}  " + "".join(f"{water_value[t][h]:>8.1f}" for h in model.plants)
        lines.append(row)
    return "\n".join(lines)


def run_model(data, mode, iters):
    print(f"\n=== Training {mode} SDDP ({SOLVER_NAME}) ===")
    m = SDDP(data, mode)
    lb = m.train(iters=iters)
    exp_cost, wv = m.simulate(n=500)
    gap = 100 * (exp_cost - lb) / exp_cost if exp_cost else 0.0
    print(f"    lower bound = {lb:.2f}   simulated cost = {exp_cost:.2f}   "
          f"gap = {gap:.1f}% (a few tenths of a % is Monte-Carlo noise)")
    print(_fmt_wv_table(m, wv))
    return m, lb, exp_cost, wv


def write_results(out_dir, results):
    """results: list of (model, lb, exp_cost, wv). Writes a CSV + JSON."""
    import csv
    import json
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "water_values.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "stage", "plant", "water_value_per_MWh"])
        for m, lb, cost, wv in results:
            for t in range(m.T):
                for h in m.plants:
                    w.writerow([m.mode, t, h, round(wv[t][h], 3)])
    summary = {
        m.mode: {
            "lower_bound": round(lb, 2),
            "expected_cost": round(cost, 2),
            "avg_water_value": {
                h: round(sum(wv[t][h] for t in range(m.T)) / m.T, 2) for h in m.plants
            },
        }
        for m, lb, cost, wv in results
    }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Wrote {csv_path} and {json_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, os.pardir, "data", "inputs_v1.json"))
    data = data_io.validate(data_io.load(path))

    random.seed(12345)
    ind, ind_lb, ind_cost, ind_wv = run_model(data, "independent", iters=40)
    random.seed(12345)
    mkv, mkv_lb, mkv_cost, mkv_wv = run_model(data, "markov", iters=40)

    # Headline comparison: same physical system, two uncertainty assumptions.
    print("\n" + "=" * 70)
    print("  COMPARISON -- same system, different uncertainty assumption")
    print("=" * 70)
    print(f"  expected total cost:  independent = {ind_cost:8.1f}   "
          f"markov = {mkv_cost:8.1f}")
    print(f"\n  Average water value over all stages ($/MWh stored):")
    print(f"    {'plant':<6}{'independent':>14}{'markov':>10}{'  difference':>14}")
    for h in ind.plants:
        a = sum(ind_wv[t][h] for t in range(ind.T)) / ind.T
        b = sum(mkv_wv[t][h] for t in range(mkv.T)) / mkv.T
        print(f"    {h:<6}{a:>14.1f}{b:>10.1f}{b - a:>14.1f}")
    print("\n  Same reservoir, same week, different number -- because the water")
    print("  value reflects the assumed future, not a physical constant.")

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(here, os.pardir, "results"))
    write_results(out_dir, [(ind, ind_lb, ind_cost, ind_wv),
                            (mkv, mkv_lb, mkv_cost, mkv_wv)])


if __name__ == "__main__":
    main()
