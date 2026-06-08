"""
generate_static_graphs.py
=========================
Generates all 5 static publication-quality graphs (white background).
Mirrors generate_figures.jl exactly but runs in pure Python/NumPy.

Usage
-----
    python generate_static_graphs.py

Outputs saved to ./figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)

RC = {
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": "#222222", "axes.linewidth": 0.9,
    "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 2.0,
}
plt.rcParams.update(RC)

# ── physics kernels ───────────────────────────────────────────────────────────

def vicsek_step(x, y, theta, L, v0, R, eta, rng):
    x = (x + v0 * np.cos(theta)) % L
    y = (y + v0 * np.sin(theta)) % L
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dx -= L * np.round(dx / L)
    dy -= L * np.round(dy / L)
    mask = dx**2 + dy**2 <= R**2
    theta = np.arctan2(mask @ np.sin(theta),
                        mask @ np.cos(theta)) + (rng.random(len(x)) - 0.5) * eta
    return x, y, theta


def run_metric(N, alpha, gamma, xi, v0, dt, T, R, seed=42, box=20.0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, box, N); y = rng.uniform(0, box, N)
    ang = rng.uniform(0, 2*np.pi, N)
    vx = v0 * np.cos(ang); vy = v0 * np.sin(ang)
    avg_start = int(0.8 * T); phi_sum = 0.0; n = 0
    for t in range(T):
        fx = np.zeros(N); fy = np.zeros(N)
        for i in range(N):
            dx = x - x[i]; dy = y - y[i]
            mask = dx**2 + dy**2 <= R**2; Nin = mask.sum()
            if Nin > 0:
                fx[i] = alpha*(vx[mask].mean()-vx[i]) - gamma*vx[i]
                fy[i] = alpha*(vy[mask].mean()-vy[i]) - gamma*vy[i]
            else:
                fx[i] = -gamma*vx[i]; fy[i] = -gamma*vy[i]
            fx[i] += xi * rng.randn(); fy[i] += xi * rng.randn()
        vx += fx*dt; vy += fy*dt; x += vx*dt; y += vy*dt
        if t >= avg_start:
            sp = np.maximum(np.sqrt(vx**2+vy**2), 1e-12)
            phi_sum += abs(np.mean(vx/sp) + 1j*np.mean(vy/sp)); n += 1
    return phi_sum / n if n > 0 else 0.0


def run_topo(N, alpha, gamma, xi, v0, dt, T, Nr, seed=42, box=20.0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, box, N); y = rng.uniform(0, box, N)
    ang = rng.uniform(0, 2*np.pi, N)
    vx = v0 * np.cos(ang); vy = v0 * np.sin(ang)
    avg_start = int(0.8 * T); phi_sum = 0.0; n = 0
    for t in range(T):
        fx = np.zeros(N); fy = np.zeros(N)
        for i in range(N):
            d2 = (x-x[i])**2 + (y-y[i])**2
            nb = np.argsort(d2)[1:Nr+1]
            fx[i] = alpha*(vx[nb].mean()-vx[i]) - gamma*vx[i] + xi*rng.randn()
            fy[i] = alpha*(vy[nb].mean()-vy[i]) - gamma*vy[i] + xi*rng.randn()
        vx += fx*dt; vy += fy*dt; x += vx*dt; y += vy*dt
        if t >= avg_start:
            sp = np.maximum(np.sqrt(vx**2+vy**2), 1e-12)
            phi_sum += abs(np.mean(vx/sp) + 1j*np.mean(vy/sp)); n += 1
    return phi_sum / n if n > 0 else 0.0


def swarm_step(x, y, xp, yp, alive, R_int, a, b, c, d, dt):
    N = alive.sum()
    if N == 0: return x, y, xp, yp, alive
    xa = x[alive]; ya = y[alive]
    dx = xa[:, None] - xa[None, :]; dy = ya[:, None] - ya[None, :]
    d2 = dx**2 + dy**2; np.fill_diagonal(d2, np.inf)
    if R_int > 0:
        mask = (d2 < R_int**2) & (d2 > 1e-10)
        Nin = mask.sum(1, keepdims=True).clip(min=1)
        sd2 = np.where(mask, d2, 1.0)
        fppx = np.where(mask, b*(-dx)-a*(-dx)/sd2, 0.0).sum(1) / Nin[:,0]
        fppy = np.where(mask, b*(-dy)-a*(-dy)/sd2, 0.0).sum(1) / Nin[:,0]
    else:
        fppx = np.zeros(N); fppy = np.zeros(N)
    dpx = xp - xa; dpy = yp - ya
    dp2 = np.maximum(dpx**2 + dpy**2, 1e-10)
    x[alive] += (fppx - c*dpx/dp2) * dt
    y[alive] += (fppy - c*dpy/dp2) * dt
    dxi = xa - xp; dyi = ya - yp
    di = np.maximum(np.sqrt(dxi**2 + dyi**2), 1e-10)
    xp += d/N * np.sum(dxi/di**3) * dt
    yp += d/N * np.sum(dyi/di**3) * dt
    alive[(x-xp)**2 + (y-yp)**2 < 0.01**2] = False
    return x, y, xp, yp, alive


def inertial_step(x,y,vx,vy,xp,yp,vxp,vyp,alive,ap,bp,gp,dp,Mpr,Mpd,dt):
    N = alive.sum()
    if N == 0: return x,y,vx,vy,xp,yp,vxp,vyp,alive
    xa=x[alive]; ya=y[alive]
    dx=xa[:,None]-xa[None,:]; dy=ya[:,None]-ya[None,:]
    d2=dx**2+dy**2; np.fill_diagonal(d2,np.inf)
    sd2=np.where(d2>1e-10,d2,1.0)
    fppx=(ap*dx/sd2-bp*dx).sum(1)/N; fppy=(ap*dy/sd2-bp*dy).sum(1)/N
    dpx=xa-xp; dpy=ya-yp; dp2=np.maximum(dpx**2+dpy**2,1e-10)
    vx[alive]+=(-vx[alive]+fppx+gp*dpx/dp2)/Mpr*dt
    vy[alive]+=(-vy[alive]+fppy+gp*dpy/dp2)/Mpr*dt
    x[alive]+=vx[alive]*dt; y[alive]+=vy[alive]*dt
    dxi=xp-xa; dyi=yp-ya; di=np.maximum(np.sqrt(dxi**2+dyi**2),1e-10)
    vxp+=(-vxp-dp*np.mean(dxi/di**3))/Mpd*dt
    vyp+=(-vyp-dp*np.mean(dyi/di**3))/Mpd*dt
    xp+=vxp*dt; yp+=vyp*dt
    kill=(x-xp)**2+(y-yp)**2<0.01**2
    alive[kill]=False; vx[kill]=0; vy[kill]=0
    return x,y,vx,vy,xp,yp,vxp,vyp,alive


# ── Static 1: phi vs eta ──────────────────────────────────────────────────────
print("[1/5] phi vs eta...")
eta_vals = np.linspace(2.5, 0.02, 50)
phi_vals = []
rng_s = np.random.RandomState(7)
xs = rng_s.uniform(0, 7, 300); ys = rng_s.uniform(0, 7, 300)
ths = rng_s.uniform(0, 2*np.pi, 300)
for ev in eta_vals:
    for _ in range(30):
        xs, ys, ths = vicsek_step(xs, ys, ths, 7.0, 0.03, 1.0, ev, rng_s)
    phi_vals.append(float(np.abs(np.mean(np.exp(1j*ths)))))

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.plot(eta_vals, phi_vals, "o-", color="#1a5ea8", ms=5)
ax.fill_between(eta_vals, phi_vals, alpha=0.12, color="#1a5ea8")
ax.axhline(0.5, color="#cc4444", lw=1, ls="--", label="phi=0.5")
ax.set_xlabel("Noise eta"); ax.set_ylabel("Order Parameter phi")
ax.set_title("Phase Transition: Disorder to Order\n(N=300, L=7, R=1, v0=0.03)")
ax.invert_xaxis(); ax.set_xlim(2.55, 0.0); ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig("figures/static1_phi_vs_eta.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved static1_phi_vs_eta.png")

# ── Static 2: metric vs topological ──────────────────────────────────────────
print("[2/5] metric vs topological...")
Ns = [50, 100, 200]; clrs = ["#1a5ea8", "#cc6600", "#228833"]
R_vals = np.arange(1.0, 13.0, 1.5); Nr_vals = np.arange(1, 15, 1)

fig, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 4.5))
for idx, N in enumerate(Ns):
    mr = [run_metric(N,1.5,0.5,0.05,1.0,0.01,600,R) for R in R_vals]
    tr = [run_topo(N,1.5,0.5,0.05,0.3,0.01,600,int(Nr)) for Nr in Nr_vals]
    ax3a.plot(R_vals, mr, "-o", color=clrs[idx], label=f"N={N}", ms=5)
    ax3b.plot(Nr_vals, tr, "-o", color=clrs[idx], label=f"N={N}", ms=5)
    print(f"  N={N} done")
for ax, title, xlabel in [
    (ax3a, "(a) Metric Interaction", "Interaction Radius R"),
    (ax3b, "(b) Topological Interaction", "Nr (number of neighbours)")]:
    ax.set_title(title); ax.set_xlabel(xlabel)
    ax.set_ylabel("Order Parameter phi")
    ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
fig.suptitle("Flocking: Metric vs Topological Interactions\n"
             "Kumar and De (2021) R Soc Open Sci 8:58",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("figures/static2_metric_vs_topo.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved static2_metric_vs_topo.png")

# ── Static 3: survival vs R_int ───────────────────────────────────────────────
print("[3/5] survival vs R_int...")
R_scan = np.arange(0.0, 2.55, 0.15)
delta_list = [0.8, 1.2, 1.8]; clrs3 = ["#1a5ea8", "#cc6600", "#cc2222"]
mks = ["o", "s", "^"]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.axvspan(0.35, 1.35, alpha=0.08, color="green")
ax.text(0.75, 133, "Optimal zone", ha="center", fontsize=8, color="#226622")
for idx, delta in enumerate(delta_list):
    Nsur = []
    for R_int in R_scan:
        tots = []
        for s in range(3):
            rng=np.random.RandomState(s*7+1)
            x=rng.random(150); y=rng.random(150)
            xp=1.2; yp=0.5; alive=np.ones(150,dtype=bool)
            for _ in range(5000):
                x,y,xp,yp,alive=swarm_step(x,y,xp,yp,alive,R_int,0.5,0.8,0.1,delta,0.005)
            tots.append(alive.sum())
        Nsur.append(np.mean(tots))
    ax.plot(R_scan, Nsur, f"-{mks[idx]}", color=clrs3[idx],
            label=f"delta={delta}", ms=6, lw=2)
    print(f"  delta={delta} done")
ax.set_xlabel("Cooperative Interaction Radius R_int")
ax.set_ylabel("Number of Survived Prey N_sur")
ax.set_title("Survival vs Interaction Range  (N=150)\n"
             "Chakraborty, Bhunia and De (2020) Sci Rep 10:8362")
ax.legend(fontsize=9); ax.grid(True, alpha=0.4); ax.set_ylim(0, 158)
fig.tight_layout()
fig.savefig("figures/static3_survival_vs_rint.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved static3_survival_vs_rint.png")

# ── Static 4: survival vs mass ────────────────────────────────────────────────
print("[4/5] survival vs mass...")
M_pd_scan = [0.1,0.3,1.0,3.0,10.0,30.0,100.0]
M_pr_list = [0.1,1.0,10.0,100.0]
clrs4 = ["#1a5ea8","#cc6600","#228833","#882299"]; mks4 = ["o","D","s","^"]


def sim_inertial_final(N, M_pr, M_pd, delta, steps=400, seed=42):
    rng=np.random.RandomState(seed)
    x=rng.random(N); y=rng.random(N); vx=np.zeros(N); vy=np.zeros(N)
    xp=1.5; yp=0.5; vxp=0.0; vyp=0.0; alive=np.ones(N,dtype=bool)
    dt=0.02
    for _ in range(steps):
        x,y,vx,vy,xp,yp,vxp,vyp,alive=inertial_step(
            x,y,vx,vy,xp,yp,vxp,vyp,alive,1.,1.,0.2,delta,M_pr,M_pd,dt)
    return alive.sum()


N_sur_mat = np.zeros((len(M_pr_list), len(M_pd_scan)))
for j, M_pr in enumerate(M_pr_list):
    for k, M_pd in enumerate(M_pd_scan):
        tots=[sim_inertial_final(80,M_pr,M_pd,2.5,seed=s*13+1) for s in range(3)]
        N_sur_mat[j,k]=np.mean(tots)
    print(f"  M_pr={M_pr} done")

ratio_vals = np.logspace(-3, 4, 25)
Nsur_ratio = np.array([
    np.mean([sim_inertial_final(80,1.0,r,2.5,seed=s*7+3) for s in range(3)])
    for r in ratio_vals])

fig, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(11, 4.8))
for j, M_pr in enumerate(M_pr_list):
    ax6a.semilogx(M_pd_scan, N_sur_mat[j], f"-{mks4[j]}",
                  color=clrs4[j], label=f"M_pr={M_pr}", ms=7, lw=2)
ax6a.set_xlabel("Predator Mass M_pd"); ax6a.set_ylabel("N survived")
ax6a.set_title("(a) N_sur vs M_pd  (N=80, delta'=2.5)")
ax6a.legend(fontsize=9); ax6a.grid(True, alpha=0.4)

ax6b.axvspan(1e-3, 0.4,  alpha=0.10, color="#cc3333", label="Killed")
ax6b.axvspan(0.4,  10.0, alpha=0.10, color="#dd8800", label="Competitive")
ax6b.axvspan(10.0, 1e4,  alpha=0.10, color="#228833", label="Survival")
ax6b.semilogx(ratio_vals, Nsur_ratio, "-o", color="#1a1a66", ms=5, lw=2)
ax6b.text(0.005, 0.92*Nsur_ratio.max(), "Killed",
          color="#aa1111", fontsize=10, fontweight="bold")
ax6b.text(0.8,   0.92*Nsur_ratio.max(), "Competitive",
          color="#996600", fontsize=9, fontweight="bold")
ax6b.text(50,    0.92*Nsur_ratio.max(), "Survival",
          color="#116611", fontsize=10, fontweight="bold")
ax6b.set_xlabel("M_pd / M_pr"); ax6b.set_ylabel("N survived")
ax6b.set_title("(b) N_sur vs M_pd/M_pr  (three regimes)")
ax6b.legend(fontsize=9, loc="center right"); ax6b.grid(True, alpha=0.4)

fig.suptitle("Inertial Model: Survival Analysis\n"
             "Chakraborty, Laha and De (2022) arXiv:2208.12280",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("figures/static4_survival_vs_mass.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved static4_survival_vs_mass.png")

# ── Static 5: phi(T) time series ─────────────────────────────────────────────
print("[5/5] phi(T) timeseries...")
Mpd_cases = [0.1, 1.0, 3.0, 100.0]
labels5 = ["(a) M_pd=0.1  light: captures all",
           "(b) M_pd=1.0  medium: splitting",
           "(c) M_pd=3.0  heavier: slower chase",
           "(d) M_pd=100  heavy: F-maneuver / escape"]
clrs5 = ["#cc2222", "#cc6600", "#1a5ea8", "#228833"]

fig, axes5 = plt.subplots(2, 2, figsize=(10, 7)); axes5 = axes5.ravel()
for k, Mpd in enumerate(Mpd_cases):
    rng=np.random.RandomState(42); N=60
    x=rng.random(N); y=rng.random(N); vx=np.zeros(N); vy=np.zeros(N)
    xp=1.5; yp=0.5+0.05*rng.randn(); vxp=0.0; vyp=0.0
    alive=np.ones(N,dtype=bool); dt=0.02; phi_s=[]; T_s=[]
    phi_int=max(1,int(1.0/dt))
    for step in range(int(300/dt)):
        x,y,vx,vy,xp,yp,vxp,vyp,alive=inertial_step(
            x,y,vx,vy,xp,yp,vxp,vyp,alive,1.,1.,0.2,2.5,1.,Mpd,dt)
        if step%phi_int==0 and alive.sum()>0:
            vpm=np.sqrt(vxp**2+vyp**2)
            if vpm>1e-12:
                vmag=np.maximum(np.sqrt(vx[alive]**2+vy[alive]**2),1e-12)
                phi=np.clip(np.mean((vx[alive]*vxp+vy[alive]*vyp)/(vmag*vpm)),-1,1)
            else: phi=0.0
            phi_s.append(phi); T_s.append(step*dt)
    ax=axes5[k]
    ax.plot(T_s, phi_s, color=clrs5[k], lw=1.8)
    ax.fill_between(T_s, phi_s, 0,
                    where=np.array(phi_s)>0, alpha=0.10, color="green")
    ax.fill_between(T_s, phi_s, 0,
                    where=np.array(phi_s)<0, alpha=0.10, color="red")
    ax.axhline(0, color="#333333", lw=0.8, ls="--")
    ax.axhline(1,  color="#cccccc", lw=0.5, ls=":")
    ax.axhline(-1, color="#cccccc", lw=0.5, ls=":")
    ax.set_ylim(-1.25, 1.25); ax.set_title(labels5[k], fontsize=9.5)
    ax.set_xlabel("Time T"); ax.set_ylabel("phi"); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.93, "Moving together",
            transform=ax.transAxes, fontsize=7.5, color="#226622", va="top")
    ax.text(0.02, 0.07, "Moving apart",
            transform=ax.transAxes, fontsize=7.5, color="#882222", va="bottom")
    print(f"  M_pd={Mpd} done")

fig.suptitle("Order Parameter phi(T): Prey-Predator Direction Alignment\n"
             "Chakraborty, Laha and De (2022) Fig 5",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("figures/static5_phi_timeseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved static5_phi_timeseries.png")

print("\n" + "="*60)
print("All static graphs saved to ./figures/")
print("="*60)
