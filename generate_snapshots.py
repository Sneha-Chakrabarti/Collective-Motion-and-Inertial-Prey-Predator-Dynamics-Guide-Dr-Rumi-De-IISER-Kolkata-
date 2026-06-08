"""
generate_snapshots.py
=====================
Generates three publication-quality snapshot figures showing
pattern formation across parameter sweeps.

  snap1_swarm_Rint_sweep.png    -- overdamped swarm, rows = R_int values
  snap2_inertial_weak_Mpd_sweep.png  -- weak predator, rows = M_pd values
  snap3_inertial_strong_Mpd_sweep.png -- strong predator, rows = M_pd values

Usage
-----
    python generate_snapshots.py

Timestamps are chosen per-row from prior diagnostic runs to show the
clearest pattern (not uniform across rows).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)

PREY = "#1a5ea8"; DEAD = "#999999"; PRED = "#cc2222"
VEL  = "#336699"; TCLR = "#cc4400"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0", "grid.linewidth": 0.4,
    "xtick.direction": "in", "ytick.direction": "in",
})

# ── physics ───────────────────────────────────────────────────────────────────

def swarm_step(x, y, xp, yp, alive, R_int, a, b, c, d, dt):
    N = alive.sum()
    if N == 0: return x, y, xp, yp, alive
    xa = x[alive]; ya = y[alive]
    dx = xa[:, None]-xa[None, :]; dy = ya[:, None]-ya[None, :]
    d2 = dx**2+dy**2; np.fill_diagonal(d2, np.inf)
    if R_int > 0:
        mask = (d2 < R_int**2) & (d2 > 1e-10)
        Nin = mask.sum(1, keepdims=True).clip(min=1)
        sd2 = np.where(mask, d2, 1.0)
        fppx = np.where(mask, b*(-dx)-a*(-dx)/sd2, 0.).sum(1)/Nin[:,0]
        fppy = np.where(mask, b*(-dy)-a*(-dy)/sd2, 0.).sum(1)/Nin[:,0]
    else:
        fppx = np.zeros(N); fppy = np.zeros(N)
    dpx = xp-xa; dpy = yp-ya; dp2 = np.maximum(dpx**2+dpy**2, 1e-10)
    x[alive] += (fppx-c*dpx/dp2)*dt; y[alive] += (fppy-c*dpy/dp2)*dt
    dxi = xa-xp; dyi = ya-yp; di = np.maximum(np.sqrt(dxi**2+dyi**2), 1e-10)
    xp += d/N*np.sum(dxi/di**3)*dt; yp += d/N*np.sum(dyi/di**3)*dt
    alive[(x-xp)**2+(y-yp)**2 < 0.01**2] = False
    return x, y, xp, yp, alive


def inertial_step(x,y,vx,vy,xp,yp,vxp,vyp,alive,ap,bp,gp,dp,Mpr,Mpd,dt):
    N = alive.sum()
    if N == 0: return x,y,vx,vy,xp,yp,vxp,vyp,alive
    xa=x[alive]; ya=y[alive]
    dx=xa[:,None]-xa[None,:]; dy=ya[:,None]-ya[None,:]
    d2=dx**2+dy**2; np.fill_diagonal(d2,np.inf)
    sd2=np.where(d2>1e-10,d2,1.)
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

# ── shared panel drawing ──────────────────────────────────────────────────────

def ax_lims(pts_x, pts_y, pad=0.12):
    xm = (pts_x.min()+pts_x.max())/2
    ym = (pts_y.min()+pts_y.max())/2
    h  = max(float(pts_x.max()-pts_x.min()),
             float(pts_y.max()-pts_y.min()))/2
    h  = max(h, 0.35)+pad
    return xm-h, xm+h, ym-h, ym+h


def setup(ax):
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.20, lw=0.4)
    ax.tick_params(direction="in", labelsize=7, length=2.5, width=0.6)
    for sp in ax.spines.values(): sp.set_linewidth(0.7)


def stamp(ax, n, T):
    ax.text(0.03, 0.97, f"T={T}  N={n}", transform=ax.transAxes,
            fontsize=7.5, va="top", color="#111111",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="#cccccc", alpha=0.85, lw=0.4))


def draw_swarm_panel(ax, x, y, xp, yp, alive, T, R_int=0):
    if alive.any():
        pts_x=np.append(x[alive],xp); pts_y=np.append(y[alive],yp)
    else:
        pts_x=np.array([xp]); pts_y=np.array([yp])
    xl,xr,yl,yr = ax_lims(pts_x, pts_y)
    ax.set_xlim(xl,xr); ax.set_ylim(yl,yr)
    if (~alive).any():
        ax.scatter(x[~alive],y[~alive],c=DEAD,s=5,edgecolors="none",
                   zorder=2,alpha=0.45)
    if alive.any():
        ax.scatter(x[alive],y[alive],c=PREY,s=6,edgecolors="none",zorder=3)
    if R_int > 0:
        ax.add_patch(Circle((xp,yp),R_int,fill=False,
                             color="#888888",lw=0.85,ls="--",alpha=0.6))
    ax.scatter([xp],[yp],c=PRED,s=90,zorder=8,
               edgecolors="#880000",linewidths=0.8,marker="o")
    stamp(ax, alive.sum(), T)


def draw_inertial_panel(ax, x,y,vx,vy,xp,yp,alive, T,
                         trail=None, quiver_scale=0.035):
    if alive.any():
        pts_x=np.append(x[alive],xp); pts_y=np.append(y[alive],yp)
    else:
        pts_x=np.array([xp]); pts_y=np.array([yp])
    if trail is not None and len(trail)>1:
        pts_x=np.append(pts_x,trail[:,0]); pts_y=np.append(pts_y,trail[:,1])
    xl,xr,yl,yr = ax_lims(pts_x, pts_y, pad=0.18)
    ax.set_xlim(xl,xr); ax.set_ylim(yl,yr)
    if trail is not None and len(trail)>1:
        n=len(trail); alp=np.linspace(0.05,0.50,n-1)
        for i in range(n-1):
            ax.plot(trail[i:i+2,0],trail[i:i+2,1],
                    color=TCLR,lw=1.0,alpha=float(alp[i]),
                    solid_capstyle="round")
    if (~alive).any():
        ax.scatter(x[~alive],y[~alive],c=DEAD,s=5,edgecolors="none",
                   zorder=2,alpha=0.45)
    if alive.any():
        ax.scatter(x[alive],y[alive],c=PREY,s=6,edgecolors="none",zorder=3)
        norm=np.maximum(np.sqrt(vx[alive]**2+vy[alive]**2),1e-12)
        sc=2*max(xr-xl,yr-yl)*quiver_scale
        ax.quiver(x[alive],y[alive],
                  vx[alive]/norm*sc,vy[alive]/norm*sc,
                  color=VEL,alpha=0.55,scale_units="xy",scale=1,
                  width=0.0025,headwidth=3.5,headlength=3.5)
    ax.scatter([xp],[yp],c=PRED,s=90,zorder=8,
               edgecolors="#880000",linewidths=0.8,marker="o")
    stamp(ax, alive.sum(), T)


# ── SNAP 1: overdamped swarm, R_int sweep ─────────────────────────────────────
# Per-row timestamps chosen from diagnostic runs.
print("Snap 1: overdamped swarm R_int sweep...")

R_cases = [
    (0.0, "R_int = 0\nNo cooperation",        [0.5, 1,  3]),
    (0.5, "R_int = 0.5\nRing around predator", [0.5, 1,  3]),
    (1.2, "R_int = 1.2\nSplitting subgroups",  [0.5, 2,  5]),
    (2.0, "R_int = 2.0\nCohesive chasing",     [0.5, 1,  2]),
]

fig = plt.figure(figsize=(9, 12)); fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.10, wspace=0.08,
                        left=0.15, right=0.97, top=0.93, bottom=0.07)
fig.text(0.56, 0.975,
    "Prey Swarm: Escape Patterns vs Interaction Radius\n"
    "Chakraborty, Bhunia and De (2020)  Sci Rep 10:8362\n"
    "N=200, b=0.8, a=0.5, c=0.1, d=0.5, dt=0.005",
    ha="center", va="top", fontsize=9, fontweight="bold",
    multialignment="center")

for row, (R_int, rlabel, T_row) in enumerate(R_cases):
    rng=np.random.RandomState(42); N=200
    x=rng.random(N); y=rng.random(N)
    xp=1.2; yp=0.5; alive=np.ones(N, dtype=bool); dt=0.005; step=0
    snaps=[]
    for T_t in T_row:
        sn=int(round(T_t/dt))-step
        for _ in range(max(0,sn)):
            x,y,xp,yp,alive=swarm_step(x,y,xp,yp,alive,R_int,0.5,0.8,0.1,0.5,dt)
            step+=1
        snaps.append((x.copy(),y.copy(),xp,yp,alive.copy()))
    print(f"  R_int={R_int} done")
    for col,(T_val,snap) in enumerate(zip(T_row,snaps)):
        ax=fig.add_subplot(gs[row,col]); setup(ax)
        draw_swarm_panel(ax,*snap,T=T_val,R_int=R_int)
        if row==0:
            ax.set_title(["Initial state","Pattern forming","Pattern established"][col],
                         fontsize=8.5, fontweight="bold", pad=4)
        if col==0: ax.set_ylabel(rlabel,fontsize=8.5,fontweight="bold",labelpad=6)
        else: ax.set_ylabel("")
        if row<3: ax.set_xticklabels([])
        if col>0: ax.set_yticklabels([])

leg_h=[
    plt.scatter([],[],c=PREY,s=25,edgecolors="none",label="Prey (alive)"),
    plt.scatter([],[],c=DEAD,s=18,edgecolors="none",alpha=0.6,label="Prey (killed)"),
    plt.scatter([],[],c=PRED,s=50,edgecolors="#880000",lw=0.8,label="Predator"),
    Line2D([0],[0],color="#888888",lw=1.1,ls="--",alpha=0.7,label="R_int radius"),
]
fig.legend(handles=leg_h, loc="lower center", ncol=4, fontsize=8.5,
           frameon=True, edgecolor="#cccccc", bbox_to_anchor=(0.56,0.005),
           handletextpad=0.4, columnspacing=0.8)
fig.savefig("figures/snap1_swarm_Rint_sweep.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("  saved snap1_swarm_Rint_sweep.png")


# ── SNAP 2: inertial weak predator, M_pd sweep ────────────────────────────────
# Ring is visible at T=150 for M_pd=2; breaks for M_pd=3 by T=300.
print("Snap 2: inertial weak predator M_pd sweep...")

Mpd_weak = [
    (2.0, "M_pd = 2.0\nStable ring",        [0, 150, 300]),
    (3.0, "M_pd = 3.0\nRing destabilises",  [0, 150, 300]),
    (3.5, "M_pd = 3.5\nChasing onset",      [0,  50, 200]),
    (5.0, "M_pd = 5.0\nActive chasing",     [0,  20, 100]),
]

fig = plt.figure(figsize=(10, 12)); fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.10, wspace=0.08,
                        left=0.17, right=0.97, top=0.93, bottom=0.07)
fig.text(0.57, 0.975,
    "Weak Predator (delta'=0.4, M_pr=1): Ring Formation to Chasing vs M_pd\n"
    "Chakraborty, Laha and De (2022)  arXiv:2208.12280\n"
    "N=200, alpha=beta=1, gamma=0.2, R_kill=0.01, dt=0.01",
    ha="center", va="top", fontsize=9, fontweight="bold",
    multialignment="center")

for row,(Mpd,rlabel,T_row) in enumerate(Mpd_weak):
    rng=np.random.RandomState(42); N=200
    x=rng.random(N); y=rng.random(N); vx=np.zeros(N); vy=np.zeros(N)
    xp=1.5; yp=0.5; vxp=0.; vyp=0.; alive=np.ones(N,dtype=bool)
    dt=0.01; step=0; snaps=[]
    for T_t in T_row:
        sn=int(round(T_t/dt))-step
        for _ in range(max(0,sn)):
            x,y,vx,vy,xp,yp,vxp,vyp,alive=inertial_step(
                x,y,vx,vy,xp,yp,vxp,vyp,alive,1.,1.,0.2,0.4,1.,Mpd,dt)
            step+=1
        snaps.append((x.copy(),y.copy(),vx.copy(),vy.copy(),xp,yp,alive.copy()))
    print(f"  M_pd={Mpd} done")
    for col,(T_val,snap) in enumerate(zip(T_row,snaps)):
        x,y,vx,vy,xp,yp,alive=snap
        ax=fig.add_subplot(gs[row,col]); setup(ax)
        draw_inertial_panel(ax,x,y,vx,vy,xp,yp,alive,T=T_val)
        if row==0:
            ax.set_title(["Initial state","Ring / pattern forming",
                          "Pattern established"][col],
                         fontsize=8.5, fontweight="bold", pad=4)
        if col==0: ax.set_ylabel(rlabel,fontsize=8.5,fontweight="bold",labelpad=6)
        else: ax.set_ylabel("")
        if row<3: ax.set_xticklabels([])
        if col>0: ax.set_yticklabels([])

leg_h=[
    plt.scatter([],[],c=PREY,s=25,edgecolors="none",label="Prey (alive)"),
    plt.scatter([],[],c=DEAD,s=18,edgecolors="none",alpha=0.6,label="Prey (killed)"),
    plt.scatter([],[],c=PRED,s=50,edgecolors="#880000",lw=0.8,label="Predator"),
    plt.quiver([],[],[],[],color=VEL,alpha=0.8,label="Velocity direction"),
]
fig.legend(handles=leg_h, loc="lower center", ncol=4, fontsize=8.5,
           frameon=True, edgecolor="#cccccc", bbox_to_anchor=(0.57,0.005),
           handletextpad=0.4, columnspacing=0.8)
fig.savefig("figures/snap2_inertial_weak_Mpd_sweep.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("  saved snap2_inertial_weak_Mpd_sweep.png")


# ── SNAP 3: inertial strong predator, M_pd sweep ──────────────────────────────
# Timestamps chosen so predator is inside/near the prey herd.
print("Snap 3: inertial strong predator M_pd sweep...")

Mpd_strong = [
    (0.1,  "M_pd = 0.1\nLight: agile chasing",         [0,  1,  3]),
    (1.0,  "M_pd = 1.0\nMedium: splitting/merging",     [0,  2,  5]),
    (10.,  "M_pd = 10\nHeavy: arc escape",              [0,  3,  7]),
    (100., "M_pd = 100\nVery heavy: F-maneuver",         [0,  3,  5]),
]
TRAIL_LEN = 60

fig = plt.figure(figsize=(10, 12)); fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.10, wspace=0.08,
                        left=0.18, right=0.97, top=0.93, bottom=0.07)
fig.text(0.57, 0.975,
    "Strong Predator (delta'=2.5, M_pr=1): Escape Trajectories vs M_pd\n"
    "Chakraborty, Laha and De (2022)  arXiv:2208.12280\n"
    "N=200, alpha=beta=1, gamma=0.2, R_kill=0.01, dt=0.02",
    ha="center", va="top", fontsize=9, fontweight="bold",
    multialignment="center")

for row,(Mpd,rlabel,T_row) in enumerate(Mpd_strong):
    rng=np.random.RandomState(99); N=200
    x=rng.random(N); y=rng.random(N); vx=np.zeros(N); vy=np.zeros(N)
    xp=1.5; yp=0.5; vxp=0.; vyp=0.; alive=np.ones(N,dtype=bool)
    dt=0.02; step=0; trail_hist=[(xp,yp)]; snaps=[]
    for T_t in T_row:
        sn=int(round(T_t/dt))-step
        for _ in range(max(0,sn)):
            x,y,vx,vy,xp,yp,vxp,vyp,alive=inertial_step(
                x,y,vx,vy,xp,yp,vxp,vyp,alive,1.,1.,0.2,2.5,1.,Mpd,dt)
            trail_hist.append((xp,yp)); step+=1
        trail=np.array(trail_hist[-TRAIL_LEN:])
        snaps.append((x.copy(),y.copy(),vx.copy(),vy.copy(),
                      xp,yp,alive.copy(),trail.copy()))
    print(f"  M_pd={Mpd} done")
    for col,(T_val,snap) in enumerate(zip(T_row,snaps)):
        x,y,vx,vy,xp,yp,alive,trail=snap
        ax=fig.add_subplot(gs[row,col]); setup(ax)
        draw_inertial_panel(ax,x,y,vx,vy,xp,yp,alive,T=T_val,
                             trail=trail if col>0 else None)
        if row==0:
            ax.set_title(["Initial state","Predator inside herd",
                          "Escape pattern"][col],
                         fontsize=8.5, fontweight="bold", pad=4)
        if col==0: ax.set_ylabel(rlabel,fontsize=8.5,fontweight="bold",labelpad=6)
        else: ax.set_ylabel("")
        if row<3: ax.set_xticklabels([])
        if col>0: ax.set_yticklabels([])

leg_h=[
    plt.scatter([],[],c=PREY,s=25,edgecolors="none",label="Prey (alive)"),
    plt.scatter([],[],c=DEAD,s=18,edgecolors="none",alpha=0.6,label="Prey (killed)"),
    plt.scatter([],[],c=PRED,s=50,edgecolors="#880000",lw=0.8,label="Predator"),
    plt.quiver([],[],[],[],color=VEL,alpha=0.8,label="Velocity direction"),
    Line2D([0],[0],color=TCLR,lw=1.4,alpha=0.6,label="Predator trajectory"),
]
fig.legend(handles=leg_h, loc="lower center", ncol=5, fontsize=8.5,
           frameon=True, edgecolor="#cccccc", bbox_to_anchor=(0.57,0.005),
           handletextpad=0.4, columnspacing=0.7)
fig.savefig("figures/snap3_inertial_strong_Mpd_sweep.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("  saved snap3_inertial_strong_Mpd_sweep.png")

print("\n" + "="*60)
print("All snapshot figures saved to ./figures/")
print("="*60)
