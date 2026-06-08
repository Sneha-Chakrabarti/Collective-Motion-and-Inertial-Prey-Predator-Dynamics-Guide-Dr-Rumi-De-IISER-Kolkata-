"""
inertial_prey_predator.jl
=========================
Inertial prey-predator model with scaled masses M_pr and M_pd.

Dimensionless equations of motion (Chakraborty, Laha & De 2022, Eqs 6-7):

  M_pr * d^2 R_i/dT^2 = -dR_i/dT
      + (1/N_sur) * sum_j [ alpha'*(R_i-R_j)/|R_i-R_j|^2 - beta'*(R_i-R_j) ]
      + gamma'*(R_i-R_p)/|R_i-R_p|^2

  M_pd * d^2 R_p/dT^2 = -dR_p/dT
      - (delta'/N_sur) * sum_i (R_p-R_i)/|R_p-R_i|^3

Fixed: alpha'=beta'=1, gamma'=0.2, R_kill=0.01.
Varied: M_pr, M_pd, delta'.

Reference
---------
Chakraborty D, Laha A, De R (2022) arXiv:2208.12280.
"""

module InertialPreyPredator

using Random, Statistics, LinearAlgebra

export InertialParams, default_params, run_inertial, compute_phi,
       mass_scan, mass_ratio_scan

struct InertialParams
    N      ::Int
    alpha  ::Float64   # prey-prey short-range repulsion
    beta   ::Float64   # prey-prey long-range attraction
    gamma  ::Float64   # prey-predator repulsion
    delta  ::Float64   # predator strength
    M_pr   ::Float64   # scaled prey mass
    M_pd   ::Float64   # scaled predator mass
    R_kill ::Float64   # kill radius
    dt     ::Float64
    T_max  ::Float64   # total simulation time
end

"""Default parameters from paper: alpha=beta=1, gamma=0.2, R_kill=0.01."""
function default_params(; N=200, M_pr=1.0, M_pd=1.0, delta=2.5, T_max=2000.0)
    InertialParams(N, 1.0, 1.0, 0.2, delta, M_pr, M_pd, 0.01, 0.02, T_max)
end

"""
    compute_phi(vx, vy, vxp, vyp, alive) -> Float64

Order parameter phi = (1/N_sur) * sum_i cos(theta_i)
where theta_i is angle between prey i velocity and predator velocity.
"""
function compute_phi(vx::Vector{Float64}, vy::Vector{Float64},
                     vxp::Float64, vyp::Float64,
                     alive::BitVector)::Float64
    N_sur = count(alive)
    N_sur == 0 && return 0.0
    vpm = sqrt(vxp^2 + vyp^2)
    vpm < 1e-12 && return 0.0
    vmag = max.(sqrt.(vx[alive].^2 .+ vy[alive].^2), 1e-12)
    cosvals = (vx[alive] .* vxp .+ vy[alive] .* vyp) ./ (vmag .* vpm)
    return clamp(mean(cosvals), -1.0, 1.0)
end

"""Single Euler step of the inertial model."""
function step!(x::Vector{Float64}, y::Vector{Float64},
               vx::Vector{Float64}, vy::Vector{Float64},
               xp::Ref{Float64}, yp::Ref{Float64},
               vxp::Ref{Float64}, vyp::Ref{Float64},
               alive::BitVector, p::InertialParams)
    N_sur = count(alive)
    N_sur == 0 && return

    xa = x[alive]; ya = y[alive]
    dt = p.dt

    # prey-prey pairwise: alpha*(R_i-R_j)/|...|^2 - beta*(R_i-R_j)
    dx  = xa .- xa'; dy  = ya .- ya'
    d2  = dx.^2 .+ dy.^2; d2[diagind(d2)] .= Inf
    sd2 = max.(d2, 1e-10)
    fppx = vec(sum(p.alpha .* dx ./ sd2 .- p.beta .* dx, dims=2)) ./ N_sur
    fppy = vec(sum(p.alpha .* dy ./ sd2 .- p.beta .* dy, dims=2)) ./ N_sur

    # prey-predator repulsion: gamma*(R_i-R_p)/|R_i-R_p|^2
    dpx = xa .- xp[]; dpy = ya .- yp[]
    dp2 = max.(dpx.^2 .+ dpy.^2, 1e-10)
    fpdx = p.gamma .* dpx ./ dp2
    fpdy = p.gamma .* dpy ./ dp2

    # prey acceleration: (-v + F_pp + F_pd) / M_pr
    ax = (-vx[alive] .+ fppx .+ fpdx) ./ p.M_pr
    ay = (-vy[alive] .+ fppy .+ fpdy) ./ p.M_pr

    vx[alive] .+= ax .* dt; vy[alive] .+= ay .* dt
    x[alive]  .+= vx[alive] .* dt; y[alive] .+= vy[alive] .* dt

    # predator force: -(delta/N_sur) * sum (R_p-R_i)/|R_p-R_i|^3
    dxi = xp[] .- xa; dyi = yp[] .- ya
    di  = max.(sqrt.(dxi.^2 .+ dyi.^2), 1e-10)
    fpx = -p.delta * mean(dxi ./ di.^3)
    fpy = -p.delta * mean(dyi ./ di.^3)

    axp = (-vxp[] + fpx) / p.M_pd
    ayp = (-vyp[] + fpy) / p.M_pd
    vxp[] += axp * dt; vyp[] += ayp * dt
    xp[]  += vxp[] * dt; yp[] += vyp[] * dt

    # kill check
    kill = (x .- xp[]).^2 .+ (y .- yp[]).^2 .< p.R_kill^2
    alive[kill] .= false
    vx[kill] .= 0.0; vy[kill] .= 0.0
end

"""
    run_inertial(p; seed, save_phi) -> (x, y, xp, yp, alive, N_survived, phi_series)

Run the inertial model for p.T_max time units.
If save_phi=true, phi is recorded every unit of time.
"""
function run_inertial(p::InertialParams; seed::Int=42, save_phi::Bool=true)
    rng = MersenneTwister(seed)
    x   = rand(rng, p.N); y = rand(rng, p.N)
    vx  = zeros(p.N);     vy = zeros(p.N)
    xp  = Ref(1.5); yp = Ref(0.5 + 0.05*randn(rng))
    vxp = Ref(0.0); vyp = Ref(0.0)
    alive = trues(p.N)

    T_steps    = round(Int, p.T_max / p.dt)
    phi_interval = max(1, round(Int, 1.0 / p.dt))
    phi_series = Float64[]

    for step in 1:T_steps
        count(alive) == 0 && break
        step!(x, y, vx, vy, xp, yp, vxp, vyp, alive, p)
        if save_phi && mod(step, phi_interval) == 0
            push!(phi_series, compute_phi(vx, vy, vxp[], vyp[], alive))
        end
    end

    return x, y, xp[], yp[], alive, count(alive), phi_series
end

"""
    mass_scan(N, delta, M_pd_values, M_pr_values; T_max, n_seeds)

Reproduce Fig 6a: N_survived vs M_pd for multiple M_pr.
"""
function mass_scan(N::Int, delta::Float64,
                   M_pd_values::Vector{Float64},
                   M_pr_values::Vector{Float64};
                   T_max::Float64=2000.0, n_seeds::Int=3)
    result = zeros(length(M_pr_values), length(M_pd_values))
    for (j, M_pr) in enumerate(M_pr_values)
        for (k, M_pd) in enumerate(M_pd_values)
            total = 0.0
            for s in 1:n_seeds
                p = default_params(N=N, M_pr=M_pr, M_pd=M_pd,
                                   delta=delta, T_max=T_max)
                _, _, _, _, _, n, _ = run_inertial(p; seed=s*13, save_phi=false)
                total += n
            end
            result[j, k] = total / n_seeds
            @info "M_pr=$M_pr  M_pd=$M_pd  N_sur=$(round(result[j,k],digits=1))"
        end
    end
    return result
end

"""
    mass_ratio_scan(N, delta, ratio_values; M_pr_base, T_max, n_seeds)

Reproduce Fig 6b: N_survived vs M_pd/M_pr.
"""
function mass_ratio_scan(N::Int, delta::Float64,
                          ratio_values::Vector{Float64};
                          M_pr_base::Float64=1.0,
                          T_max::Float64=2000.0, n_seeds::Int=3)
    N_sur = zeros(length(ratio_values))
    for (k, r) in enumerate(ratio_values)
        M_pd = r * M_pr_base
        total = 0.0
        for s in 1:n_seeds
            p = default_params(N=N, M_pr=M_pr_base, M_pd=M_pd,
                               delta=delta, T_max=T_max)
            _, _, _, _, _, n, _ = run_inertial(p; seed=s*7, save_phi=false)
            total += n
        end
        N_sur[k] = total / n_seeds
    end
    return N_sur
end

end # module
