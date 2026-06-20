# sddp.jl -- Julia port of src/sddp.py.
#
# A small, transparent LP-based SDDP engine for Small SDDP. One engine, three
# "modes" of belief about the future:
#
#   :independent  -- stagewise-independent (memoryless, marginal probabilities)
#   :markov       -- stagewise-dependent, exogenous Markov chain (drought persistence)
#   :dry_only     -- independent but restricted to the 5 driest scenarios
#                    (a pessimist's truncated belief). Built via active_scenarios=...
#
# All three optionally carry an end-of-horizon SALVAGE VALUE: piecewise-linear
# marginal water value with prescribed values at (min, initial, max) storage,
# integrated into a convex end-cost function and installed as cuts on stage T.
# That cost propagates back through the SDDP recursion so water values do not
# collapse to zero at the end of the horizon.
#
# Run:  julia --project=. src/sddp.jl
#       (or `using Pkg; Pkg.add(["JuMP","HiGHS","JSON"])` then `include("src/sddp.jl")`)

using JuMP, HiGHS, JSON, Random, Printf

# -----------------------------------------------------------------------------
# A Benders cut on the future-cost variable theta of a stage subproblem:
#     theta >= intercept + sum_h slope[h] * storage_end[h]
# -----------------------------------------------------------------------------
struct Cut
    intercept::Float64
    slope::Dict{String,Float64}
end

# -----------------------------------------------------------------------------
# An SDDP model carries the physical system (immutable), the uncertainty tables
# and chosen probability structure, plus a per-(stage, regime) pool of cuts.
# -----------------------------------------------------------------------------
mutable struct SDDP
    mode::Symbol                                # :independent or :markov
    T::Int                                      # number of stages
    S::Int                                      # number of scenarios / Markov states
    df::Float64                                 # one-stage discount factor 1/(1+r)

    plants::Vector{String}                      # ordered plant names
    hydro::Dict{String,Dict{String,Any}}        # plant -> parameters
    cap::Float64                                # thermal capacity (MWh/stage)
    voll::Float64                               # unserved energy cost ($/MWh)
    load::Vector{Float64}                       # demand[t]
    x0::Dict{String,Float64}                    # initial storage

    inflow::Dict{String,Vector{Vector{Float64}}}  # inflow[h][t][m]
    tcost::Vector{Vector{Float64}}              # tcost[t][m]

    initial::Vector{Float64}                    # initial distribution over states
    P::Vector{Vector{Float64}}                  # transition matrix P[m][n]

    # CUTS ARE INDEXED BY (t, m) -- READ BEFORE EDITING.
    # There is one value function V_t^m(x) per (stage, Markov state). cuts[t][m]
    # are the Benders cuts for V_t^m, and the subproblem at (t, m) must use that
    # state's own data inflow[h][t][m] / tcost[t][m] together with cuts[t][m].
    # Regime matching is enforced by THREE structural things, NOT by the cut algebra:
    #   1. this per-(t,m) cut pool,
    #   2. each successor subproblem at (t+1, n) using regime-n data + cuts, and
    #   3. the parent weighting successors by ITS OWN row P[m] in `backward!`.
    # In :independent mode every P[m] equals the marginal probabilities, so the
    # cuts built for every m are identical -- the pools collapse to a single value
    # function per stage. In :markov mode the rows differ and each regime needs its
    # own V_t^m. DO NOT collapse cuts to a single per-stage pool to "simplify": it
    # would apply dry-successor cuts to wet parents and silently break Markov.
    cuts::Vector{Vector{Vector{Cut}}}            # cuts[t][m] :: Vector{Cut}

    lb_history::Vector{Float64}
end

# -----------------------------------------------------------------------------
# Constructor. `mode` is :independent or :markov. `active_scenarios` restricts the
# (independent-mode) belief to a subset of scenario indices, renormalised -- this
# is how we build the dry-only pessimist. `end_value_points` is the salvage curve
# (mv_min, mv_init, mv_max), the MARGINAL water value at min / initial / max
# storage; if given it is installed as cuts on the last stage.
# -----------------------------------------------------------------------------
function SDDP(data; mode::Symbol,
              active_scenarios::Union{Nothing,Vector{Int}}=nothing,
              end_value_points::Union{Nothing,Tuple{Real,Real,Real}}=nothing)
    @assert mode in (:independent, :markov)
    u = data["uncertainty"]
    T = data["meta"]["stages"]
    S = data["meta"]["scenarios"]
    r = data["meta"]["discount_rate_per_stage"]
    df = 1.0 / (1.0 + r)

    plants = String[h["name"] for h in data["hydro"]]
    hydro = Dict{String,Dict{String,Any}}(h["name"] => h for h in data["hydro"])
    thermal = data["thermal"][1]
    therm_name = thermal["name"]
    cap = float(thermal["capacity_MWh_per_stage"])
    voll = float(data["demand"]["unserved_energy_cost_per_MWh"])
    load = float.(data["demand"]["load_MWh_per_stage"])
    x0 = Dict{String,Float64}(h["name"] => float(h["initial_storage_MWh"]) for h in data["hydro"])

    # inflow[h][t][m] :: Float64 -- copy out of JSON for type stability
    inflow = Dict{String,Vector{Vector{Float64}}}()
    for h in plants
        inflow[h] = [float.(u["inflows_MWh"][h][t]) for t in 1:T]
    end
    tcost = [float.(u["thermal_cost_per_MWh"][therm_name][t]) for t in 1:T]

    # Initial distribution and transition rows.
    if mode === :markov
        active_scenarios === nothing || error("active_scenarios only applies to :independent")
        mk = u["markov"]
        initial = float.(mk["initial_distribution"])
        P = [float.(row) for row in mk["transition_matrix"]]
    else
        p = float.(u["scenario_probabilities"])
        if active_scenarios === nothing
            w = copy(p)
        else
            active = Set(active_scenarios)
            masked = [i in active ? p[i] : 0.0 for i in 1:S]
            tot = sum(masked)
            tot > 0 || error("active_scenarios has total probability 0")
            w = masked ./ tot                 # renormalise over believed scenarios
        end
        initial = w
        P = [copy(w) for _ in 1:S]           # rows all equal -> memoryless
    end

    cuts = [[Cut[] for _ in 1:S] for _ in 1:T]
    model = SDDP(mode, T, S, df, plants, hydro, cap, voll, load, x0,
                 inflow, tcost, initial, P, cuts, Float64[])

    end_value_points === nothing || install_end_value!(model, end_value_points)
    return model
end

# -----------------------------------------------------------------------------
# End-of-horizon salvage value (the V-ENDVALUE lever in PLAN.md).
#
# `points` = (mv_min, mv_init, mv_max): the MARGINAL water value ($/MWh) at each
# plant's own min-storage, initial-storage, and max-storage levels, linearly
# interpolated. The marginal is decreasing, so the integrated salvage COST
#     EC_h(s) = integral_s^Umax  mv_h(u) du
# is convex and decreasing -- exactly what a set of Benders cuts represents. The
# value function is separable across plants, so a joint cut is the sum of per-
# plant tangents; we add one cut per fill-fraction along the [min..max] diagonal.
#
# These cuts sit in cuts[T][m] for every m and -- because the backward pass only
# writes stages < T -- they persist and propagate backward through the whole
# recursion, lifting water values at every stage.
# -----------------------------------------------------------------------------
function install_end_value!(m::SDDP, points::Tuple{Real,Real,Real}; n_breaks::Int=21)
    mv_min, mv_init, mv_max = float.(points)
    bounds = Dict(h => (float(m.hydro[h]["min_storage_MWh"]),
                        float(m.hydro[h]["initial_storage_MWh"]),
                        float(m.hydro[h]["max_storage_MWh"])) for h in m.plants)

    mv(h, s) = begin
        L, I, U = bounds[h]
        s <= L && return mv_min
        s >= U && return mv_max
        s <= I ? mv_min + (mv_init - mv_min) * (s - L) / (I - L) :
                 mv_init + (mv_max - mv_init) * (s - I) / (U - I)
    end

    # Integrated salvage cost = trapezoidal area under mv from p up to full.
    EC(h, p) = begin
        L, I, U = bounds[h]
        p = clamp(p, L, U)
        if p >= I
            return 0.5 * (mv(h, p) + mv_max) * (U - p)
        end
        area_low  = 0.5 * (mv(h, p) + mv_init) * (I - p)
        area_high = 0.5 * (mv_init + mv_max)   * (U - I)
        return area_low + area_high
    end

    salvage_cuts = Cut[]
    for k in 0:(n_breaks - 1)
        lam = k / (n_breaks - 1)              # fill fraction min..max
        slope = Dict{String,Float64}()
        intercept = 0.0
        for h in m.plants
            L, I, U = bounds[h]
            p = L + lam * (U - L)
            m_hp = mv(h, p)
            slope[h] = -m_hp                  # EC'(p) = -mv(p)
            intercept += EC(h, p) + m_hp * p  # tangent of EC_h at p
        end
        push!(salvage_cuts, Cut(intercept, slope))
    end
    for s in 1:m.S
        m.cuts[m.T][s] = copy(salvage_cuts)
    end
    return m
end

# -----------------------------------------------------------------------------
# Stage subproblem at (t, mm) with incoming storage x. Builds and solves the LP
# with JuMP+HiGHS, then reads the cost, end-of-stage storage, immediate cost
# (for the upper bound), and the duals of the storage-FIXING constraints.
#
# WATER VALUE = -dual(Fix[h]). The dual of `s0[h] == x[h]` is d(cost)/d(x[h]) <= 0;
# we report its negation -- the cost saved per extra MWh stored.
# -----------------------------------------------------------------------------
function subproblem(M::SDDP, t::Int, mm::Int, x::Dict{String,Float64})
    lp = Model(HiGHS.Optimizer)
    set_silent(lp)

    H = M.plants
    @variable(lp, s0[h in H])                                              # free; pinned by Fix
    @variable(lp, M.hydro[h]["min_storage_MWh"] <= s1[h in H] <= M.hydro[h]["max_storage_MWh"])
    @variable(lp, 0 <= rel[h in H] <= M.hydro[h]["max_turbine_MWh_per_stage"])
    @variable(lp, spl[h in H] >= 0)
    @variable(lp, 0 <= gen <= M.cap)
    @variable(lp, uns >= 0)
    @variable(lp, theta >= 0)                                              # end value >= 0

    # Storage-fixing constraints -- their duals are the incoming water values.
    @constraint(lp, Fix[h in H], s0[h] == x[h])

    # Water balance (no cascade in v1): s1 = s0 + inflow - release - spill
    @constraint(lp, [h in H], s1[h] == s0[h] + M.inflow[h][t][mm] - rel[h] - spl[h])

    # Demand balance: hydro energy + thermal + unserved == load
    @constraint(lp,
        sum(M.hydro[h]["production_factor_MWh_per_MWh"] * rel[h] for h in H)
        + gen + uns == M.load[t])

    # Benders cuts on the future cost. (At t = T these include the salvage cuts
    # installed by install_end_value!; otherwise they were added by backward!.)
    for cut in M.cuts[t][mm]
        @constraint(lp, theta >= cut.intercept + sum(cut.slope[h] * s1[h] for h in H))
    end

    # Objective: immediate variable cost + discounted future cost.
    @objective(lp, Min, M.tcost[t][mm] * gen + M.voll * uns + M.df * theta)
    optimize!(lp)

    storage_end = Dict{String,Float64}(h => value(s1[h]) for h in H)
    fix_dual    = Dict{String,Float64}(h => dual(Fix[h]) for h in H)
    immediate   = M.tcost[t][mm] * value(gen) + M.voll * value(uns)
    return objective_value(lp), immediate, storage_end, fix_dual
end

# -----------------------------------------------------------------------------
# Random sampling helpers.
# -----------------------------------------------------------------------------
function sample_index(probs::Vector{Float64}, rng::AbstractRNG)
    u = rand(rng); c = 0.0
    for (i, p) in enumerate(probs)
        c += p
        u < c && return i
    end
    return length(probs)
end

sample_initial(M::SDDP, rng::AbstractRNG) = sample_index(M.initial, rng)
sample_next(M::SDDP, m::Int, rng::AbstractRNG) = sample_index(M.P[m], rng)

# -----------------------------------------------------------------------------
# Forward pass: one simulated path. Returns the per-stage end-of-stage storage
# (trial points for the backward pass) and the discounted immediate cost.
# -----------------------------------------------------------------------------
function forward!(M::SDDP, rng::AbstractRNG)
    x = copy(M.x0)
    m = sample_initial(M, rng)
    trial = Vector{Dict{String,Float64}}(undef, M.T)
    cost = 0.0
    for t in 1:M.T
        _, immediate, s_end, _ = subproblem(M, t, m, x)
        cost += M.df^(t-1) * immediate
        trial[t] = s_end
        x = s_end
        t < M.T && (m = sample_next(M, m, rng))
    end
    return trial, cost
end

# -----------------------------------------------------------------------------
# Backward pass. At stage t the cut approximates the expected value of being at
# end-of-stage-t storage y = trial[t], evaluated via the successor subproblems
# at stage t+1. A P[m]-weighted cut is added to every node (t, m).
#
# Regime matching is enforced HERE by two of the three structural points (see
# the comment on SDDP.cuts): (2) each successor at (t+1, n) is solved with its
# own regime-n data and cut pool, and (3) the parent (t, m) weights those
# successors by its OWN transition row P[m]. The single-cut algebra is then a
# valid lower-support of the convex V_{t+1}^n's at y in BOTH modes.
# -----------------------------------------------------------------------------
function backward!(M::SDDP, trial::Vector{Dict{String,Float64}})
    # Stop at T-1: the last stage already holds salvage cuts (if any) and is
    # never written by the backward pass.
    for t in (M.T - 1):-1:1
        y = trial[t]
        # Solve every successor state once at the trial point.
        Q = Vector{Float64}(undef, M.S)
        g = Vector{Dict{String,Float64}}(undef, M.S)
        for n in 1:M.S
            obj_n, _, _, fix_n = subproblem(M, t + 1, n, y)
            Q[n] = obj_n
            g[n] = fix_n   # d(obj_n)/d(incoming storage) = cut gradient
        end
        # Build a P[m]-weighted cut for each node (t, m):
        #   theta_(t,m) >= sum_n P[m][n] * ( Q_n + g_n . (s1 - y) )
        for mm in 1:M.S
            w = M.P[mm]
            slope = Dict{String,Float64}(
                h => sum(w[n] * g[n][h] for n in 1:M.S) for h in M.plants)
            intercept = sum(w[n] * (Q[n] - sum(g[n][h] * y[h] for h in M.plants)) for n in 1:M.S)
            push!(M.cuts[t][mm], Cut(intercept, slope))
        end
    end
end

# -----------------------------------------------------------------------------
# Deterministic lower bound: expected first-stage value over the initial
# distribution, using the current cuts. A valid lower bound on E[cost].
# -----------------------------------------------------------------------------
function lower_bound(M::SDDP)
    lb = 0.0
    for m in 1:M.S
        M.initial[m] == 0 && continue
        obj, _, _, _ = subproblem(M, 1, m, M.x0)
        lb += M.initial[m] * obj
    end
    return lb
end

# -----------------------------------------------------------------------------
# Training loop: each iteration runs `n_paths` forward simulations (better
# exploration of low-storage states) and a backward pass, then recomputes the
# lower bound. Stops on a relative-change tolerance.
# -----------------------------------------------------------------------------
function train!(M::SDDP; iters::Int=40, n_paths::Int=5,
                tol::Float64=1e-3, patience::Int=5,
                rng::AbstractRNG=MersenneTwister(12345), verbose::Bool=true)
    stable = 0
    for it in 1:iters
        for _ in 1:n_paths
            trial, _ = forward!(M, rng)
            backward!(M, trial)
        end
        lb = lower_bound(M)
        push!(M.lb_history, lb)
        if verbose && (it <= 3 || it % 5 == 0)
            @printf("    iter %3d   lower bound = %10.2f\n", it, lb)
        end
        if length(M.lb_history) >= 2
            prev = M.lb_history[end-1]
            if prev > 0 && abs(lb - prev) / abs(prev) < tol
                stable += 1
                if stable >= patience
                    verbose && @printf("    converged after %d iterations (LB = %.2f).\n", it, lb)
                    break
                end
            else
                stable = 0
            end
        end
    end
    return isempty(M.lb_history) ? 0.0 : M.lb_history[end]
end

# -----------------------------------------------------------------------------
# Monte-Carlo simulation under the trained policy. Returns mean total cost and
# the average water value per (stage, plant) over the sampled paths.
# Water value = -fix_dual (cost saved per extra stored MWh).
# -----------------------------------------------------------------------------
function simulate(M::SDDP; n::Int=300, seed::Int=0)
    rng = MersenneTwister(seed)
    total = 0.0
    wv_sum = [Dict(h => 0.0 for h in M.plants) for _ in 1:M.T]
    counts = zeros(Int, M.T)
    for _ in 1:n
        x = copy(M.x0)
        m = sample_initial(M, rng)
        cost = 0.0
        for t in 1:M.T
            _, immediate, s_end, fix_dual = subproblem(M, t, m, x)
            cost += M.df^(t-1) * immediate
            for h in M.plants
                wv_sum[t][h] += -fix_dual[h]
            end
            counts[t] += 1
            x = s_end
            t < M.T && (m = sample_next(M, m, rng))
        end
        total += cost
    end
    exp_cost = total / n
    water_value = [Dict(h => wv_sum[t][h] / counts[t] for h in M.plants) for t in 1:M.T]
    return exp_cost, water_value
end

# -----------------------------------------------------------------------------
# Pretty-print helpers (mirrors the Python CLI output).
# -----------------------------------------------------------------------------
function format_wv_table(label::String, M::SDDP, water_value)
    io = IOBuffer()
    println(io, "  WATER VALUES (\$/MWh stored) -- $label model")
    print(io, "    stage  "); for h in M.plants; print(io, lpad(h, 8)); end; println(io)
    for t in 1:M.T
        print(io, "    ", lpad(t, 5), "  ")
        for h in M.plants
            print(io, lpad(@sprintf("%.1f", water_value[t][h]), 8))
        end
        println(io)
    end
    return String(take!(io))
end

function run_model(data, label::String, iters::Int; kwargs...)
    println("\n=== Training $label SDDP (HiGHS) ===")
    M = SDDP(data; kwargs...)
    lb = train!(M; iters=iters)
    exp_cost, wv = simulate(M; n=500)
    gap = exp_cost == 0 ? 0.0 : 100 * (exp_cost - lb) / exp_cost
    @printf("    lower bound = %.2f   simulated cost = %.2f   gap = %.1f%%\n",
            lb, exp_cost, gap)
    print(format_wv_table(label, M, wv))
    return (label, M, lb, exp_cost, wv)
end

# -----------------------------------------------------------------------------
# Main: same three-way comparison as the Python `sddp.py main()`, all carrying
# the end-of-horizon salvage value (1000 $/MWh at min, 35 at initial, 0 at max).
# -----------------------------------------------------------------------------
function main()
    here = @__DIR__
    path = joinpath(here, "..", "data", "inputs_v1.json") |> normpath
    data = JSON.parsefile(path)

    DRY_SCENARIOS = [1, 2, 3, 4, 5]               # 1-indexed in Julia
    END_VALUE = (1000.0, 35.0, 0.0)
    specs = [
        ("independent", (mode=:independent, end_value_points=END_VALUE)),
        ("markov",      (mode=:markov,      end_value_points=END_VALUE)),
        ("dry-only",    (mode=:independent, active_scenarios=DRY_SCENARIOS,
                          end_value_points=END_VALUE)),
    ]
    results = [run_model(data, label, 40; nt...) for (label, nt) in specs]

    println("\n", "="^70)
    println("  COMPARISON -- same system, different beliefs about the future")
    println("="^70)
    println("  expected total cost (model's own belief):")
    for (label, M, lb, cost, wv) in results
        @printf("    %-14s = %8.1f\n", label, cost)
    end

    plants = results[1][2].plants
    println("\n  Average water value over all stages (\$/MWh stored):")
    print("    ", lpad("plant", 6))
    for (label, _, _, _, _) in results; print(lpad(label, 14)); end; println()
    for h in plants
        print("    ", rpad(h, 6))
        for (_, M, _, _, wv) in results
            avg = sum(wv[t][h] for t in 1:M.T) / M.T
            print(lpad(@sprintf("%.1f", avg), 14))
        end
        println()
    end
    println("\n  Same reservoir, same week, three different numbers -- because the")
    println("  water value reflects the assumed future, not a physical constant.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
