"""
flocking_interactions.jl
========================
Metric and topological flocking model from Kumar and De (2021).

Equation of motion:
    dv_i/dt = (alpha/N_in) * sum_{N_in}(v_j - v_i) - gamma*v_i + xi

Reference
---------
Kumar V and De R (2021) Efficient flocking: metric versus topological
interactions. R Soc Open Sci 8:58.

Reviewed in
-----------
De R and Chakraborty D (2022) J Biosci 47:48, Eq (3).
"""

module FlockingInteractions

using Random, Statistics, LinearAlgebra

export run_metric_flocking, run_topological_flocking, phi_vect

"""
    phi_vect(vx, vy) -> Float64

Vectorial order parameter phi = (1/N)|sum_i v_hat_i|.
"""
function phi_vect(vx::Vector{Float64}, vy::Vector{Float64})::Float64
    speeds = sqrt.(vx.^2 .+ vy.^2)
    speeds[speeds .< 1e-12] .= 1e-12
    return abs(sum(vx ./ speeds .+ im .* vy ./ speeds)) / length(vx)
end

"""
    run_metric_flocking(N, alpha, gamma, xi_sigma, v0, dt, T_steps, R; seed, box_size)

Metric flocking: each particle aligns with neighbours within radius R.
Returns the time-averaged phi over the last 20% of steps.
"""
function run_metric_flocking(N::Int, alpha::Float64, gamma::Float64,
                              xi_sigma::Float64, v0::Float64,
                              dt::Float64, T_steps::Int, R::Float64;
                              seed::Int=42, box_size::Float64=20.0)::Float64
    rng = MersenneTwister(seed)
    x   = rand(rng, N) .* box_size
    y   = rand(rng, N) .* box_size
    ang = rand(rng, N) .* 2pi
    vx  = v0 .* cos.(ang)
    vy  = v0 .* sin.(ang)

    avg_start = round(Int, 0.8 * T_steps)
    phi_sum = 0.0; n_avg = 0

    for t in 1:T_steps
        fx = zeros(N); fy = zeros(N)
        for i in 1:N
            dx = x .- x[i]; dy = y .- y[i]
            d2 = dx.^2 .+ dy.^2
            nb = findall(d2 .<= R^2)
            Nin = length(nb)
            if Nin > 0
                fx[i] = alpha*(mean(vx[nb]) - vx[i]) - gamma*vx[i]
                fy[i] = alpha*(mean(vy[nb]) - vy[i]) - gamma*vy[i]
            else
                fx[i] = -gamma*vx[i]; fy[i] = -gamma*vy[i]
            end
            fx[i] += xi_sigma * randn(rng)
            fy[i] += xi_sigma * randn(rng)
        end
        vx .+= fx .* dt; vy .+= fy .* dt
        x  .+= vx .* dt; y  .+= vy .* dt
        if t >= avg_start
            phi_sum += phi_vect(vx, vy); n_avg += 1
        end
    end
    return n_avg > 0 ? phi_sum / n_avg : 0.0
end

"""
    run_topological_flocking(N, alpha, gamma, xi_sigma, v0, dt, T_steps, Nr; seed, box_size)

Topological flocking: each particle aligns with its Nr nearest neighbours.
Returns the time-averaged phi over the last 20% of steps.
"""
function run_topological_flocking(N::Int, alpha::Float64, gamma::Float64,
                                   xi_sigma::Float64, v0::Float64,
                                   dt::Float64, T_steps::Int, Nr::Int;
                                   seed::Int=42, box_size::Float64=20.0)::Float64
    rng = MersenneTwister(seed)
    x   = rand(rng, N) .* box_size
    y   = rand(rng, N) .* box_size
    ang = rand(rng, N) .* 2pi
    vx  = v0 .* cos.(ang)
    vy  = v0 .* sin.(ang)

    avg_start = round(Int, 0.8 * T_steps)
    phi_sum = 0.0; n_avg = 0

    for t in 1:T_steps
        fx = zeros(N); fy = zeros(N)
        for i in 1:N
            dx = x .- x[i]; dy = y .- y[i]
            d2 = dx.^2 .+ dy.^2
            sorted = sortperm(d2)
            nb = sorted[2:min(Nr+1, N)]   # exclude self (index 1)
            if length(nb) > 0
                fx[i] = alpha*(mean(vx[nb]) - vx[i]) - gamma*vx[i]
                fy[i] = alpha*(mean(vy[nb]) - vy[i]) - gamma*vy[i]
            else
                fx[i] = -gamma*vx[i]; fy[i] = -gamma*vy[i]
            end
            fx[i] += xi_sigma * randn(rng)
            fy[i] += xi_sigma * randn(rng)
        end
        vx .+= fx .* dt; vy .+= fy .* dt
        x  .+= vx .* dt; y  .+= vy .* dt
        if t >= avg_start
            phi_sum += phi_vect(vx, vy); n_avg += 1
        end
    end
    return n_avg > 0 ? phi_sum / n_avg : 0.0
end

end # module
