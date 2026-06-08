"""
prey_predator_swarm.jl
======================
Overdamped prey-predator model (mu = 1, zero inertia).

Equations of motion (De & Chakraborty 2022, Eqs 4-5):

  dr_i/dt = f_i^{prey-prey}(R_int) + f_i^{prey-pred}

  f_i^{pp} = (1/N_in) * sum_{j in R_int} [b*(r_j-r_i) - a*(r_j-r_i)/|r_j-r_i|^2]

  f_i^{pred} = -c*(r_p-r_i)/|r_p-r_i|^2   (prey repelled from predator)

  dr_p/dt = (d/N) * sum_i (r_i-r_p)/|r_i-r_p|^3

Reference
---------
Chakraborty D, Bhunia S, De R (2020) Sci Rep 10:8362.
De R and Chakraborty D (2022) J Biosci 47:48.
"""

module PreyPredatorSwarm

using Random, Statistics, LinearAlgebra

export SwarmParams, run_swarm, survival_scan

struct SwarmParams
    N      ::Int
    b      ::Float64   # prey-prey attraction
    a      ::Float64   # prey-prey repulsion
    c      ::Float64   # prey-predator repulsion
    d      ::Float64   # predator strength
    R_kill ::Float64   # kill radius
    dt     ::Float64
    T_max  ::Int
end

"""Default parameters from Chakraborty et al. (2020)."""
default_params(; N=200, d=0.5) = SwarmParams(N, 0.8, 0.5, 0.1, d, 0.01, 0.005, 10000)

"""Single Euler step of the overdamped swarm model."""
function step!(x::Vector{Float64}, y::Vector{Float64},
               xp::Ref{Float64}, yp::Ref{Float64},
               alive::BitVector,
               R_int::Float64, p::SwarmParams)
    N_alive = count(alive)
    N_alive == 0 && return

    xa = x[alive]; ya = y[alive]

    # prey-prey pairwise forces
    dx = xa .- xa'; dy = ya .- ya'
    d2 = dx.^2 .+ dy.^2
    d2[diagind(d2)] .= Inf

    fppx = zeros(N_alive); fppy = zeros(N_alive)
    if R_int > 0
        mask = (d2 .< R_int^2) .& (d2 .> 1e-10)
        Nin  = max.(sum(mask, dims=2)[:, 1], 1)
        sd2  = ifelse.(mask, d2, ones(size(d2)))
        fppx = vec(sum(ifelse.(mask, p.b .* (-dx) .- p.a .* (-dx) ./ sd2, 0.0), dims=2) ./ Nin)
        fppy = vec(sum(ifelse.(mask, p.b .* (-dy) .- p.a .* (-dy) ./ sd2, 0.0), dims=2) ./ Nin)
    end

    # prey-predator repulsion
    dpx = xp[] .- xa; dpy = yp[] .- ya
    dp2 = max.(dpx.^2 .+ dpy.^2, 1e-10)

    x[alive] .+= (fppx .- p.c .* dpx ./ dp2) .* p.dt
    y[alive] .+= (fppy .- p.c .* dpy ./ dp2) .* p.dt

    # predator-prey attraction
    dxi = xa .- xp[]; dyi = ya .- yp[]
    di  = max.(sqrt.(dxi.^2 .+ dyi.^2), 1e-10)
    xp[] += p.d / N_alive * sum(dxi ./ di.^3) * p.dt
    yp[] += p.d / N_alive * sum(dyi ./ di.^3) * p.dt

    # kill check
    kill = (x .- xp[]).^2 .+ (y .- yp[]).^2 .< p.R_kill^2
    alive[kill] .= false
end

"""
    run_swarm(p, R_int; seed) -> (x, y, xp, yp, alive, N_survived)

Run a single simulation with interaction radius R_int.
"""
function run_swarm(p::SwarmParams, R_int::Float64; seed::Int=42)
    rng   = MersenneTwister(seed)
    x     = rand(rng, p.N)
    y     = rand(rng, p.N)
    xp    = Ref(1.2 + rand(rng)*0.1)
    yp    = Ref(0.5)
    alive = trues(p.N)

    for _ in 1:p.T_max
        count(alive) == 0 && break
        step!(x, y, xp, yp, alive, R_int, p)
    end

    return x, y, xp[], yp[], alive, count(alive)
end

"""
    survival_scan(p, R_int_values; n_seeds) -> Vector{Float64}

Compute mean N_survived over n_seeds realisations for each R_int.
"""
function survival_scan(p::SwarmParams, R_int_values::Vector{Float64};
                        n_seeds::Int=5)
    N_sur = zeros(length(R_int_values))
    for (k, R_int) in enumerate(R_int_values)
        total = 0.0
        for s in 1:n_seeds
            _, _, _, _, _, n = run_swarm(p, R_int; seed=s*7)
            total += n
        end
        N_sur[k] = total / n_seeds
        @info "R_int=$(round(R_int,digits=2))  N_sur=$(round(N_sur[k],digits=1))"
    end
    return N_sur
end

end # module
