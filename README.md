# Collective-Motion-and-Inertial-Prey-Predator-Dynamics-Guide-Dr-Rumi-De-IISER-Kolkata

# Collective Motion and Inertial Prey-Predator Dynamics

Computational reproduction of two papers from Dr. Rumi De's group at IISER Kolkata,
prepared under the supervision of Dr. Rumi De, IISER Kolkata.

**Papers reproduced**

1. De R and Chakraborty D (2022). Collective motion: influence of local behavioural
   interactions among individuals. *J Biosci* **47**:48.

2. Chakraborty D, Laha A and De R (2022). Effect of inertia on the evasion and pursuit
   dynamics of prey swarms and the emergence of an optimal mass ratio for the
   predator-prey arms race. *arXiv:2208.12280*.

---

## Animations

| File | Description |
|------|-------------|
| `gif1_vicsek_evolution.gif` | Vicsek 4-panel: disorder to order, phi bar above each panel |
| `gif2_order_parameter_transition.gif` | Phase transition: noise sweep with live phi curve |
| `gif3_swarm_escape_patterns.gif` | 4 R_int escape patterns side-by-side, dynamic axes |
| `gif4_survival_vs_rint.gif` | Survival curves building live with swarm panel |
| `gif5_weak_predator_ring.gif` | Weak predator: ring vs chasing, phi subplot |
| `gif6_strong_predator_escape.gif` | Strong predator: 3 mass cases with velocity arrows and trail |
| `gif7_phi_timeseries.gif` | phi(T) building live for 4 predator masses |
| `gif8_phase_diagram.gif` | Phase diagram M_pd sweep with live simulation |

No ffmpeg required. All animations use PillowWriter (pure Python/Pillow).

## Static Figures

| File | Description |
|------|-------------|
| `static1_phi_vs_eta.png` | Order parameter phi vs noise eta (Vicsek phase transition) |
| `static2_metric_vs_topo.png` | phi vs R (metric) and phi vs Nr (topological), N=50/100/200 |
| `static3_survival_vs_rint.png` | N_sur vs R_int for 3 predator strengths |
| `static4_survival_vs_mass.png` | N_sur vs M_pd and vs M_pd/M_pr (three regimes) |
| `static5_phi_timeseries.png` | phi(T) for 4 predator masses, full time series |

## Snapshot Figures

| File | Description |
|------|-------------|
| `snap1_swarm_Rint_sweep.png` | Overdamped swarm: rows = R_int, cols = time (per-row timestamps) |
| `snap2_inertial_weak_Mpd_sweep.png` | Weak predator: rows = M_pd, ring formation at T=150 |
| `snap3_inertial_strong_Mpd_sweep.png` | Strong predator: rows = M_pd, predator inside herd in col 2 |

---

## Repository Structure

```
.
+-- src/
|   +-- vicsek_model.jl              Julia: Vicsek (1995) SPP model
|   +-- flocking_interactions.jl     Julia: metric vs topological (Kumar & De 2021)
|   +-- prey_predator_swarm.jl       Julia: overdamped swarm (Chakraborty et al. 2020)
|   +-- inertial_prey_predator.jl    Julia: inertial model (Chakraborty et al. 2022)
+-- generate_figures.jl              Julia: run all sims, save static figures
+-- generate_static_graphs.py        Python: same static figures (no Julia needed)
+-- generate_snapshots.py            Python: snapshot parameter-sweep figures
+-- generate_animations.py           Python: all 8 GIF animations
+-- Project.toml                     Julia package manifest
+-- requirements.txt                 Python dependencies
+-- report/
|   +-- report.tex                   LaTeX source for summary report
+-- figures/                         Output: static PNGs
+-- animations/                      Output: GIF animations
```

---

## Models

### Vicsek Model (1995)

Each of N particles moves at fixed speed v0 and aligns with neighbours within radius R:

```
x_i(t + dt) = x_i(t) + v_i * dt   (mod L, periodic BC)
theta_i(t + dt) = <theta>_R + Uniform(-eta/2, eta/2)
```

Order parameter: phi = |mean(exp(i*theta))|. Phase transition from phi~0 (disordered) to
phi~1 (ordered) as eta decreases or density increases.

Parameters used: N=300, v0=0.03, R=1, eta in {0.1, 2.0}, L in {5, 7, 25}.

### Metric vs Topological Flocking (Kumar and De 2021)

```
dv_i/dt = (alpha / N_in) * sum_{N_in} (v_j - v_i) - gamma * v_i + xi
```

Metric: N_in = neighbours within radius R.
Topological: N_in = fixed Nr nearest neighbours.

### Overdamped Prey-Predator Swarm (Chakraborty et al. 2020)

Equations of motion (mu = 1, overdamped):

```
dr_i/dt = (1/N_in) * sum_{R_int} [b*(r_j - r_i) - a*(r_j - r_i)/|r_j - r_i|^2]
          - c*(r_p - r_i) / |r_p - r_i|^2

dr_p/dt = (d/N) * sum_i (r_i - r_p) / |r_i - r_p|^3
```

Key result: intermediate R_int maximises prey survival. Too small: no coordination.
Too large: cohesive group easily tracked.

### Inertial Model (Chakraborty, Laha and De 2022)

Scaled dimensionless equations with masses M_pr and M_pd:

```
M_pr * d^2 R_i / dT^2 = -dR_i/dT
    + (1/N_sur) * sum_j [alpha'*(R_i-R_j)/|R_i-R_j|^2 - beta'*(R_i-R_j)]
    + gamma'*(R_i-R_p)/|R_i-R_p|^2

M_pd * d^2 R_p / dT^2 = -dR_p/dT
    - (delta'/N_sur) * sum_i (R_p-R_i) / |R_p-R_i|^3
```

Fixed: alpha' = beta' = 1, gamma' = 0.2, R_kill = 0.01.

Three regimes identified by M_pd/M_pr:
- **Killed** (ratio less than ~0.4): agile predator captures all prey
- **Competitive** (~0.4 to ~10): arms race, some prey survive
- **Survival** (ratio greater than ~10): heavy predator cannot manoeuvre

---

## Running

### Python only (no Julia required)

```bash
pip install -r requirements.txt

# static graphs
python generate_static_graphs.py

# snapshot figures
python generate_snapshots.py

# all 8 GIF animations
python generate_animations.py
```

### Julia + Python

```bash
# install Julia packages
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# generate static figures via Julia
julia --project=. generate_figures.jl

# animations still via Python
python generate_animations.py
```

---

## Parameter Reference

| Model | Parameter | Value | Notes |
|-------|-----------|-------|-------|
| Vicsek | N | 300 | particles |
| | v0 | 0.03 | speed |
| | R | 1.0 | interaction radius |
| | eta | 0.1 or 2.0 | low / high noise |
| Flocking | alpha | 1.5 | alignment strength |
| | gamma | 0.5 | drag |
| Overdamped swarm | N | 200 | prey |
| | b | 0.8 | attraction |
| | a | 0.5 | repulsion |
| | c | 0.1 | prey-predator repulsion |
| | d | 0.5 | predator strength |
| | R_int | 0 to 2.0 | swept |
| Inertial | N | 200 | prey |
| | alpha' = beta' | 1.0 | interaction strengths |
| | gamma' | 0.2 | prey-predator coupling |
| | delta' | 0.4 or 2.5 | weak / strong predator |
| | M_pd | 0.1 to 100 | swept |
| | dt | 0.01 to 0.02 | timestep |

---

## References

1. Vicsek T et al. (1995) Phys Rev Lett 75:1226
2. Kumar V and De R (2021) R Soc Open Sci 8:58
3. Chakraborty D, Bhunia S and De R (2020) Sci Rep 10:8362
4. De R and Chakraborty D (2022) J Biosci 47:48
5. Chakraborty D, Laha A and De R (2022) arXiv:2208.12280
6. Ballerini M et al. (2008) Proc Natl Acad Sci USA 105:1232
7. Wilson AM et al. (2018) Nature 554:183

---

*IISER Kolkata, under Dr. Rumi De*
