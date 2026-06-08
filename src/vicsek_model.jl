"""
vicsek_model.jl
===============
Vectorised reproduction of the Vicsek (1995) self-propelled particle model.

Reference
---------
Vicsek T, Czirok A, Ben-Jacob E, Cohen I, Shochet O (1995)
Novel type of phase transition in a system of self-driven particles.
Phys Rev Lett 75:1226.

Reviewed in
-----------
De R and Chakraborty D (2022) J Biosci 47:48.
"""

module VicsekModel

using Random, Statistics, LinearAlgebra

export run_vicsek, order_parameter

"""
    run_vicsek(N, L, v0, R, eta, T_steps; seed=42)

Run the Vicsek model for T_steps timesteps (dt=1).

Parameters
----------
N        : number of particles
L        : box side length (periodic boundary)
v0       : constant speed of each particle
R        : neighbour interaction radius
eta      : noise amplitude (uniform in [-eta/2, eta/2])
T_steps  : number of steps to run

Returns
-------
x, y     : final positions (N,)
theta    : final headings (N,)
phi_series : order parameter at every step (T_steps,)
"""
function run_vicsek(N::Int, L::Float64, v0::Float64, R::Float64,
                    eta::Float64, T_steps::Int; seed::Int=42)
    rng = MersenneTwister(seed)

    x     = rand(rng, N) .* L
    y     = rand(rng, N) .* L
    theta = rand(rng, N) .* 2pi

    phi_series = zeros(T_steps)

    for t in 1:T_steps
        # update positions (periodic BC)
        x .= mod.(x .+ v0 .* cos.(theta), L)
        y .= mod.(y .+ v0 .* sin.(theta), L)

        # pairwise displacement with minimum image
        dx = x .- x'                       # (N,N)
        dy = y .- y'
        dx .-= L .* round.(dx ./ L)
        dy .-= L .* round.(dy ./ L)
        mask = dx.^2 .+ dy.^2 .<= R^2     # neighbour mask

        # vectorised average heading
        sum_sin = mask * sin.(theta)
        sum_cos = mask * cos.(theta)
        theta   = atan.(sum_sin, sum_cos) .+
                  (rand(rng, N) .- 0.5) .* eta

        phi_series[t] = order_parameter(theta)
    end

    return x, y, theta, phi_series
end

"""Scalar order parameter phi = |<exp(i*theta)>|."""
function order_parameter(theta::Vector{Float64})::Float64
    return abs(mean(exp.(im .* theta)))
end

end # module
