"""
generate_figures.jl
===================
Master Julia script: runs all simulations and saves static figures.

Usage
-----
    julia --project=. generate_figures.jl

Outputs saved to ./figures/
"""

include("src/vicsek_model.jl")
include("src/flocking_interactions.jl")
include("src/prey_predator_swarm.jl")
include("src/inertial_prey_predator.jl")

using .VicsekModel
using .FlockingInteractions
using .PreyPredatorSwarm
using .InertialPreyPredator
using Plots, Statistics, LinearAlgebra, DelimitedFiles
gr()
mkpath("figures")

println("="^60)
println("Generating all static figures")
println("="^60)

# ── Fig 1: phi vs eta (Vicsek phase transition) ───────────────
println("\n[1/5] phi vs eta...")
eta_vals = collect(range(2.5, 0.02, length=50))
phi_vals = Float64[]
for eta in eta_vals
    _, _, _, phi_s = run_vicsek(300, 7.0, 0.03, 1.0, eta, 500; seed=7)
    push!(phi_vals, mean(phi_s[end-50:end]))
end

p1 = plot(eta_vals, phi_vals, marker=:circle, ms=4, lw=2, color=:royalblue,
          xlabel="Noise η", ylabel="Order Parameter φ",
          title="Phase Transition: Disorder → Order\n(N=300, L=7, R=1, v₀=0.03)",
          legend=false, framestyle=:box, tickdir=:in,
          fillrange=0, fillalpha=0.12, fillcolor=:royalblue)
hline!(p1, [0.5], ls=:dash, color=:crimson, lw=1, label="φ=0.5")
xflip!(p1)
savefig(p1, "figures/static1_phi_vs_eta.png")
println("  saved static1_phi_vs_eta.png")

# ── Fig 2: metric vs topological flocking ─────────────────────
println("\n[2/5] metric vs topological...")
Ns       = [50, 100, 200]
clrs     = [:royalblue, :darkorange, :green]
R_vals   = collect(1.0:1.5:12.5)
Nr_vals  = collect(1:1:14)

p2a = plot(title="(a) Metric Interaction", xlabel="Interaction Radius R",
           ylabel="Order Parameter φ", ylims=(0,1.05),
           framestyle=:box, tickdir=:in, legend=:topleft)
p2b = plot(title="(b) Topological Interaction",
           xlabel="Nᵣ (neighbours)", ylabel="Order Parameter φ",
           ylims=(0,1.05), framestyle=:box, tickdir=:in, legend=:topleft)

for (idx, N) in enumerate(Ns)
    mr = [run_metric_flocking(N,1.5,0.5,0.05,1.0,0.01,600,R; seed=42)
          for R in R_vals]
    tr = [run_topological_flocking(N,1.5,0.5,0.05,0.3,0.01,600,Nr; seed=42)
          for Nr in Nr_vals]
    plot!(p2a, R_vals,  mr, label="N=$N", color=clrs[idx], lw=2, marker=:circle, ms=4)
    plot!(p2b, Nr_vals, tr, label="N=$N", color=clrs[idx], lw=2, marker=:circle, ms=4)
    println("  N=$N done")
end

p2 = plot(p2a, p2b, layout=(1,2), size=(900,420),
          plot_title="Flocking: Metric vs Topological Interactions")
savefig(p2, "figures/static2_metric_vs_topo.png")
println("  saved static2_metric_vs_topo.png")

# ── Fig 3: survival vs R_int ──────────────────────────────────
println("\n[3/5] survival vs R_int...")
R_scan      = collect(0.0:0.15:2.5)
delta_list  = [0.8, 1.2, 1.8]
clrs3       = [:royalblue, :darkorange, :crimson]

p3 = plot(title="Survival vs Cooperative Interaction Range\n(N=150)",
          xlabel="Interaction Radius R_int",
          ylabel="N survived", framestyle=:box, tickdir=:in, legend=:topright)
vspan!(p3, [0.35, 1.35], alpha=0.07, color=:green, label="Optimal zone")

for (idx, delta) in enumerate(delta_list)
    params = default_params(N=150, d=delta)
    params_mod = SwarmParams(150, params.b, params.a, params.c,
                              delta, params.R_kill, params.dt, 5000)
    Nsur = survival_scan(params_mod, R_scan; n_seeds=3)
    plot!(p3, R_scan, Nsur, label="δ=$delta", color=clrs3[idx],
          lw=2, marker=:circle, ms=3)
    println("  delta=$delta done")
end
savefig(p3, "figures/static3_survival_vs_rint.png")
println("  saved static3_survival_vs_rint.png")

# ── Fig 4: survival vs mass / mass ratio ──────────────────────
println("\n[4/5] survival vs mass...")
M_pd_scan   = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
M_pr_list   = [0.1, 1.0, 10.0, 100.0]
clrs4       = [:royalblue, :darkorange, :green, :purple]

N_sur_mat = mass_scan(80, 2.5, M_pd_scan, M_pr_list;
                       T_max=300.0, n_seeds=3)

p4a = plot(title="(a) N_sur vs M_pd  (N=80, δ'=2.5)",
           xlabel="M_pd", ylabel="N survived",
           xscale=:log10, framestyle=:box, tickdir=:in, legend=:topleft)
for (j, M_pr) in enumerate(M_pr_list)
    plot!(p4a, M_pd_scan, N_sur_mat[j,:],
          label="M_pr=$M_pr", color=clrs4[j], lw=2, marker=:diamond, ms=5)
end

ratio_vals  = exp10.(range(-3, 4, length=25))
N_sur_ratio = mass_ratio_scan(80, 2.5, ratio_vals;
                               M_pr_base=1.0, T_max=300.0, n_seeds=3)

p4b = plot(title="(b) N_sur vs M_pd/M_pr  (three regimes)",
           xlabel="M_pd / M_pr", ylabel="N survived",
           xscale=:log10, framestyle=:box, tickdir=:in, legend=false)
vspan!(p4b, [1e-3, 0.4],  alpha=0.12, color=:crimson,    label="Killed")
vspan!(p4b, [0.4,  10.0], alpha=0.12, color=:darkorange,  label="Competitive")
vspan!(p4b, [10.0, 1e4],  alpha=0.12, color=:green,       label="Survival")
plot!(p4b, ratio_vals, N_sur_ratio, color=:navy, lw=2, marker=:circle, ms=4)
annotate!(p4b, 0.01, 0.92*maximum(N_sur_ratio), text("Killed",      :crimson,    9, :left))
annotate!(p4b, 1.0,  0.92*maximum(N_sur_ratio), text("Competitive", :darkorange, 9, :center))
annotate!(p4b, 300,  0.92*maximum(N_sur_ratio), text("Survival",    :green,      9, :right))

p4 = plot(p4a, p4b, layout=(1,2), size=(960,440),
          plot_title="Inertial Model: Survival Analysis")
savefig(p4, "figures/static4_survival_vs_mass.png")
println("  saved static4_survival_vs_mass.png")

# ── Fig 5: phi(T) time series ─────────────────────────────────
println("\n[5/5] phi(T) time series...")
Mpd_cases = [0.1, 1.0, 3.0, 100.0]
labels5   = ["(a) M_pd=0.1  light → captures all",
             "(b) M_pd=1.0  medium → splitting",
             "(c) M_pd=3.0  heavier → slower chase",
             "(d) M_pd=100  heavy → F-maneuver"]
clrs5     = [:crimson, :darkorange, :royalblue, :green]

phi_plots = []
for (k, Mpd) in enumerate(Mpd_cases)
    p = default_params(N=60, M_pd=Mpd, delta=2.5, T_max=300.0)
    _, _, _, _, _, _, phi_s = run_inertial(p; seed=42, save_phi=true)
    T_arr = collect(range(0, 300.0, length=length(phi_s)))
    pl = plot(T_arr, clamp.(phi_s, -1, 1),
              color=clrs5[k], lw=1.8, legend=false,
              title=labels5[k], xlabel="Time T", ylabel="φ",
              ylims=(-1.25, 1.25), framestyle=:box, tickdir=:in)
    hline!(pl, [0.0], ls=:dash, color=:gray, lw=0.8)
    vspan!(pl, [0, 300], [0, 1.1],   alpha=0.05, color=:green)
    vspan!(pl, [0, 300], [-1.1, 0],  alpha=0.05, color=:red)
    push!(phi_plots, pl)
    println("  M_pd=$Mpd done")
end

p5 = plot(phi_plots..., layout=(2,2), size=(900,640),
          plot_title="Order Parameter phi(T): Prey-Predator Direction Alignment")
savefig(p5, "figures/static5_phi_timeseries.png")
println("  saved static5_phi_timeseries.png")

println("\n" * "="^60)
println("All static figures saved to ./figures/")
println("="^60)
