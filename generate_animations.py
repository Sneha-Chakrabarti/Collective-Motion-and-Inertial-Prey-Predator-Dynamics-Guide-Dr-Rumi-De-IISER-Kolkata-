"""
generate_animations.py
======================
Generates all 8 GIF animations. No ffmpeg required (PillowWriter only).

GIF 1  -- Vicsek 4-panel: disorder to order evolution
GIF 2  -- Vicsek phase transition: noise sweep with live phi curve
GIF 3  -- Prey swarm: 4 R_int escape patterns side-by-side
GIF 4  -- Prey swarm: survival curves building live
GIF 5  -- Inertial weak predator: ring vs chasing (2 panels + phi subplot)
GIF 6  -- Inertial strong predator: 3 mass cases with velocity and trail
GIF 7  -- phi(T) time series building live: 4 predator masses
GIF 8  -- Phase diagram: M_pd sweep with live simulation

Usage
-----
    python generate_animations.py

Outputs saved to ./animations/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

os.makedirs("animations", exist_ok=True)

WRITER = PillowWriter(fps=15)
PREY   = "#1a5ea8"; DEAD = "#999999"; PRED = "#cc2222"
VEL    = "#336699"; TCLR = "#cc4400"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0", "grid.linewidth": 0.4,
    "xtick.direction": "in", "ytick.direction": "in",
})


# ── vectorised physics ────────────────────────────────────────────────────────

def vicsek_step(x, y, theta, L, v0, R, eta, rng):
    x = (x + v0*np.cos(theta)) % L
    y = (y + v0*np.sin(theta)) % L
    dx = x[:,None]-x[None,:]; dy = y[:,None]-y[None,:]
    dx -= L*np.round(dx/L); dy -= L*np.round(dy/L)
    mask = dx**2+dy**2 <= R**2
    theta = np.arctan2(mask@np.sin(theta),
                        mask@np.cos(theta)) + (rng.random(len(x))-0.5)*eta
    return x, y, theta


def sim_vicsek(N,L,v0,R,eta,n_frames,spf,seed=42):
    rng=np.random.RandomState(seed)
    x=rng.uniform(0,L,N); y=rng.uniform(0,L,N)
    th=rng.uniform(0,2*np.pi,N)
    frames=[(x.copy(),y.copy(),th.copy(),
             float(np.abs(np.mean(np.exp(1j*th)))))]
    for _ in range(n_frames):
        for _ in range(spf): x,y,th=vicsek_step(x,y,th,L,v0,R,eta,rng)
        frames.append((x.copy(),y.copy(),th.copy(),
                       float(np.abs(np.mean(np.exp(1j*th))))))
    return frames


def swarm_step(x,y,xp,yp,alive,R_int,a,b,c,d,dt):
    N=alive.sum()
    if N==0: return x,y,xp,yp,alive
    xa=x[alive]; ya=y[alive]
    dx=xa[:,None]-xa[None,:]; dy=ya[:,None]-ya[None,:]
    d2=dx**2+dy**2; np.fill_diagonal(d2,np.inf)
    if R_int>0:
        mask=(d2<R_int**2)&(d2>1e-10); Nin=mask.sum(1,keepdims=True).clip(min=1)
        sd2=np.where(mask,d2,1.)
        fppx=np.where(mask,b*(-dx)-a*(-dx)/sd2,0.).sum(1)/Nin[:,0]
        fppy=np.where(mask,b*(-dy)-a*(-dy)/sd2,0.).sum(1)/Nin[:,0]
    else: fppx=np.zeros(N); fppy=np.zeros(N)
    dpx=xp-xa; dpy=yp-ya; dp2=np.maximum(dpx**2+dpy**2,1e-10)
    x[alive]+=(fppx-c*dpx/dp2)*dt; y[alive]+=(fppy-c*dpy/dp2)*dt
    dxi=xa-xp; dyi=ya-yp; di=np.maximum(np.sqrt(dxi**2+dyi**2),1e-10)
    xp+=d/N*np.sum(dxi/di**3)*dt; yp+=d/N*np.sum(dyi/di**3)*dt
    alive[(x-xp)**2+(y-yp)**2<0.01**2]=False
    return x,y,xp,yp,alive


def sim_swarm(N,R_int,n_frames,spf,dt=0.005,seed=42):
    rng=np.random.RandomState(seed)
    x=rng.random(N); y=rng.random(N); xp=1.2; yp=0.5
    alive=np.ones(N,dtype=bool)
    frames=[(x.copy(),y.copy(),xp,yp,alive.copy())]
    for _ in range(n_frames):
        for _ in range(spf):
            x,y,xp,yp,alive=swarm_step(x,y,xp,yp,alive,R_int,0.5,0.8,0.1,1.5,dt)
        frames.append((x.copy(),y.copy(),xp,yp,alive.copy()))
    return frames


def inertial_step(x,y,vx,vy,xp,yp,vxp,vyp,alive,ap,bp,gp,dp,Mpr,Mpd,dt):
    N=alive.sum()
    if N==0: return x,y,vx,vy,xp,yp,vxp,vyp,alive
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


def sim_inertial(N,ap,bp,gp,dp,Mpr,Mpd,n_frames,spf,dt=0.02,seed=42):
    rng=np.random.RandomState(seed)
    x=rng.random(N); y=rng.random(N); vx=np.zeros(N); vy=np.zeros(N)
    xp=1.5; yp=0.5+0.05*rng.randn(); vxp=0.; vyp=0.
    alive=np.ones(N,dtype=bool); phi_s=[]
    frames=[(x.copy(),y.copy(),vx.copy(),vy.copy(),xp,yp,alive.copy())]
    for _ in range(n_frames):
        for _ in range(spf):
            x,y,vx,vy,xp,yp,vxp,vyp,alive=inertial_step(
                x,y,vx,vy,xp,yp,vxp,vyp,alive,ap,bp,gp,dp,Mpr,Mpd,dt)
        N_s=alive.sum(); vpm=np.sqrt(vxp**2+vyp**2)
        if N_s>0 and vpm>1e-12:
            vmag=np.maximum(np.sqrt(vx[alive]**2+vy[alive]**2),1e-12)
            phi=np.clip(np.mean((vx[alive]*vxp+vy[alive]*vyp)/(vmag*vpm)),-1,1)
        else: phi=0.
        phi_s.append(phi)
        frames.append((x.copy(),y.copy(),vx.copy(),vy.copy(),xp,yp,alive.copy()))
    return frames, phi_s


def dyn_lims(pts_x, pts_y, pad=0.12):
    xm=(pts_x.min()+pts_x.max())/2; ym=(pts_y.min()+pts_y.max())/2
    h=max(float(pts_x.max()-pts_x.min()),float(pts_y.max()-pts_y.min()))/2
    h=max(h,0.35)+pad
    return xm-h,xm+h,ym-h,ym+h


# ── GIF 1: Vicsek 4-panel ────────────────────────────────────────────────────
print("[GIF 1] Vicsek 4-panel evolution...")
cfgs=[
    dict(N=300,L=7.0, v0=0.03,R=1.,eta=2.0,spf=5,seed=1,
         title="(a) High noise, disordered"),
    dict(N=300,L=25.0,v0=0.03,R=1.,eta=0.1,spf=5,seed=2,
         title="(b) Low density, ordered clusters"),
    dict(N=300,L=7.0, v0=0.03,R=1.,eta=2.0,spf=5,seed=3,
         title="(c) High density, high noise"),
    dict(N=300,L=5.0, v0=0.03,R=1.,eta=0.1,spf=5,seed=4,
         title="(d) High density, ordered flock"),
]
NF=160
all_v=[(c,sim_vicsek(c["N"],c["L"],c["v0"],c["R"],c["eta"],
                      NF,c["spf"],c["seed"])) for c in cfgs]

fig1,axes1=plt.subplots(2,2,figsize=(10,8.5))
fig1.subplots_adjust(wspace=0.28,hspace=0.45,left=0.07,right=0.97,top=0.90,bottom=0.06)
fig1.suptitle("Vicsek Model: Emergence of Collective Motion\n"
              "De and Chakraborty (2022)  J Biosci 47:48",
              fontsize=11,fontweight="bold",y=0.97)
axes1=axes1.ravel()
quivs=[]; ttxts=[]; phi_lines=[]; phi_hist=[[] for _ in range(4)]

for k,(cfg,frames) in enumerate(all_v):
    ax=axes1[k]
    ax.set_xlim(0,cfg["L"]); ax.set_ylim(0,cfg["L"]); ax.set_aspect("equal")
    ax.tick_params(direction="in"); ax.grid(True,alpha=0.2)
    ax.set_xlabel("x",fontsize=8); ax.set_ylabel("y",fontsize=8)
    ax.set_title(cfg["title"],fontsize=8.5,pad=3)
    x0,y0,th0,p0=frames[0]
    q=ax.quiver(x0,y0,np.cos(th0),np.sin(th0),np.abs(np.cos(th0)),
                cmap="RdYlBu_r",scale=0.9,width=0.004,headwidth=3.5,
                norm=plt.Normalize(-1,1))
    quivs.append(q)
    txt=ax.text(0.02,0.96,"T=0  phi=0.00",transform=ax.transAxes,
                color="#333333",fontsize=7.5,va="top")
    ttxts.append(txt)
    phi_ax=ax.inset_axes([0.0,1.02,1.0,0.10])
    phi_ax.set_xlim(0,NF); phi_ax.set_ylim(0,1)
    phi_ax.tick_params(labelsize=5,direction="in",colors="#666666")
    phi_ax.set_ylabel("phi",fontsize=6,color="#226622")
    ln,=phi_ax.plot([],[],color="#226622",lw=1.5); phi_lines.append(ln)
    for sp in phi_ax.spines.values(): sp.set_linewidth(0.5)

def upd1(frame):
    arts=[]
    for k,(cfg,frames) in enumerate(all_v):
        fi=min(frame,len(frames)-1); x,y,th,p=frames[fi]
        quivs[k].set_offsets(np.column_stack([x,y]))
        quivs[k].set_UVC(np.cos(th),np.sin(th),np.abs(np.cos(th)))
        ttxts[k].set_text(f"T={fi*cfg['spf']}  phi={p:.3f}")
        phi_hist[k].append(p); phi_lines[k].set_data(range(len(phi_hist[k])),phi_hist[k])
        arts+=[quivs[k],ttxts[k],phi_lines[k]]
    return arts

ani1=FuncAnimation(fig1,upd1,frames=NF+1,interval=60,blit=True)
ani1.save("animations/gif1_vicsek_evolution.gif",writer=WRITER)
plt.close(fig1)
print("  saved gif1_vicsek_evolution.gif")


# ── GIF 2: Phase transition sweep ────────────────────────────────────────────
print("[GIF 2] Phase transition noise sweep...")
eta_anim=np.linspace(2.5,0.03,90)
rng2=np.random.RandomState(7)
xa2=rng2.uniform(0,7,300); ya2=rng2.uniform(0,7,300)
tha2=rng2.uniform(0,2*np.pi,300)
phi_acc2=[]; eta_acc2=[]

fig2,(ax2a,ax2b)=plt.subplots(1,2,figsize=(10,4.5))
fig2.suptitle("Vicsek Phase Transition  (N=300, L=7, R=1)",fontsize=10,fontweight="bold")
ax2a.set_xlim(0,7); ax2a.set_ylim(0,7); ax2a.set_aspect("equal")
ax2a.set_xlabel("x"); ax2a.set_ylabel("y")
ax2a.set_title("Particle Configuration",fontsize=9); ax2a.grid(True,alpha=0.2)
ax2b.set_xlim(0.,2.6); ax2b.set_ylim(-0.02,1.05); ax2b.invert_xaxis()
ax2b.set_xlabel("Noise eta"); ax2b.set_ylabel("Order Parameter phi")
ax2b.set_title("phi vs eta  (phase transition)",fontsize=9)
ax2b.axhline(0.5,color="#cc4444",lw=0.9,ls="--",alpha=0.7)
ax2b.grid(True,alpha=0.3)

q2=ax2a.quiver(xa2,ya2,np.cos(tha2),np.sin(tha2),np.abs(np.cos(tha2)),
               cmap="RdYlBu_r",scale=25,width=0.004,headwidth=3,
               norm=plt.Normalize(-1,1))
phi_line2,=ax2b.plot([],[],"-",color="#1a5ea8",lw=2)
eta_pt2,  =ax2b.plot([],[],"o",color="#cc2222",ms=9,zorder=5)
eta_txt2  =ax2a.text(0.02,0.96,"",transform=ax2a.transAxes,
                     color="#333333",fontsize=9,va="top")
phi_txt2  =ax2a.text(0.02,0.87,"",transform=ax2a.transAxes,
                     color="#226622",fontsize=9,va="top")

def upd2(frame):
    global xa2,ya2,tha2
    ev=eta_anim[frame]
    for _ in range(15): xa2,ya2,tha2=vicsek_step(xa2,ya2,tha2,7.,0.03,1.,ev,rng2)
    pv=float(np.abs(np.mean(np.exp(1j*tha2))))
    phi_acc2.append(pv); eta_acc2.append(ev)
    q2.set_offsets(np.column_stack([xa2,ya2]))
    q2.set_UVC(np.cos(tha2),np.sin(tha2),np.abs(np.cos(tha2)))
    phi_line2.set_data(eta_acc2,phi_acc2); eta_pt2.set_data([ev],[pv])
    eta_txt2.set_text(f"eta = {ev:.3f}"); phi_txt2.set_text(f"phi = {pv:.3f}")
    return q2,phi_line2,eta_pt2,eta_txt2,phi_txt2

ani2=FuncAnimation(fig2,upd2,frames=len(eta_anim),interval=80,blit=True)
ani2.save("animations/gif2_order_parameter_transition.gif",writer=WRITER)
plt.close(fig2)
print("  saved gif2_order_parameter_transition.gif")


# ── GIF 3: Swarm escape patterns 4-panel ────────────────────────────────────
print("[GIF 3] Swarm escape patterns 4-panel...")
R_cases=[0.0,0.5,1.2,2.0]
labels3=["R_int=0.0\n(no cooperation)","R_int=0.5\n(ring around predator)",
         "R_int=1.2\n(splitting)","R_int=2.0\n(cohesive chasing)"]
NFS=200; SPFS=30
all_sw=[sim_swarm(100,R,NFS,SPFS,seed=42) for R in R_cases]

fig3,axes3=plt.subplots(1,4,figsize=(15,4.2))
fig3.suptitle("Prey Swarm Escape Patterns vs Interaction Radius\n"
              "Chakraborty, Bhunia and De (2020)",fontsize=11,fontweight="bold")
fig3.subplots_adjust(wspace=0.32,left=0.05,right=0.98,top=0.80,bottom=0.12)
sp3p=[]; sp3d=[]; sp3r=[]; sur3t=[]

for k,ax in enumerate(axes3):
    ax.set_aspect("equal"); ax.grid(True,alpha=0.2)
    ax.tick_params(direction="in"); ax.set_xlabel("x",fontsize=7.5)
    ax.set_ylabel("y",fontsize=7.5); ax.set_title(labels3[k],fontsize=8.5,pad=4)
    x0,y0,xp0,yp0,al0=all_sw[k][0]
    s1=ax.scatter(x0[al0],y0[al0],c=PREY,s=6,edgecolors="none",zorder=3)
    s2=ax.scatter([],[],c=DEAD,s=5,edgecolors="none",alpha=0.4,zorder=2)
    s3=ax.scatter([xp0],[yp0],c=PRED,s=90,zorder=5,
                  edgecolors="#880000",linewidths=0.6,marker="o")
    st=ax.text(0.03,0.97,"N=100",transform=ax.transAxes,
               color="#333333",fontsize=8,va="top")
    for sp in ax.spines.values(): sp.set_linewidth(0.7)
    sp3p.append(s1); sp3d.append(s2); sp3r.append(s3); sur3t.append(st)
ttxt3=fig3.text(0.5,0.01,"T=0.00",ha="center",color="#333333",fontsize=9)

def upd3(frame):
    arts=[]
    for k in range(4):
        fi=min(frame,len(all_sw[k])-1); x,y,xp,yp,alive=all_sw[k][fi]
        if alive.any():
            pts_x=np.append(x[alive],xp); pts_y=np.append(y[alive],yp)
            xl,xr,yl,yr=dyn_lims(pts_x,pts_y)
            axes3[k].set_xlim(xl,xr); axes3[k].set_ylim(yl,yr)
            sp3p[k].set_offsets(np.column_stack([x[alive],y[alive]]))
        else: sp3p[k].set_offsets(np.empty((0,2)))
        if (~alive).any():
            sp3d[k].set_offsets(np.column_stack([x[~alive],y[~alive]]))
        sp3r[k].set_offsets([[xp,yp]]); sur3t[k].set_text(f"N={alive.sum()}")
        arts+=[sp3p[k],sp3d[k],sp3r[k],sur3t[k]]
    ttxt3.set_text(f"T={frame*SPFS*0.005:.2f}"); arts.append(ttxt3)
    return arts

ani3=FuncAnimation(fig3,upd3,frames=NFS+1,interval=55,blit=False)
ani3.save("animations/gif3_swarm_escape_patterns.gif",writer=WRITER)
plt.close(fig3)
print("  saved gif3_swarm_escape_patterns.gif")


# ── GIF 4: Survival scan live ─────────────────────────────────────────────────
print("[GIF 4] Survival scan live build...")
R_scan4=np.arange(0.,2.55,0.18); delta_list4=[0.8,1.2,1.8]
clrs4=["#1a5ea8","#cc6600","#cc2222"]
Nsur4={}
for delta in delta_list4:
    Nsur4[delta]=[]
    for R_int in R_scan4:
        tots=[sim_swarm(120,R_int,1,4000,seed=s*7+1)[-1][4].sum() for s in range(3)]
        Nsur4[delta].append(np.mean(tots))
    print(f"  delta={delta} done")
live4=sim_swarm(120,0.8,200,20,seed=99)

fig4,(ax4a,ax4b)=plt.subplots(1,2,figsize=(11,4.5))
fig4.suptitle("Prey Survival vs Cooperative Interaction Range\n"
              "Chakraborty, Bhunia and De (2020)",fontsize=10,fontweight="bold")
ax4a.set_aspect("equal"); ax4a.grid(True,alpha=0.2)
ax4a.set_xlabel("x"); ax4a.set_ylabel("y")
ax4a.set_title("Live Swarm  (R_int=0.8, delta=1.2)",fontsize=9)
sp4p=ax4a.scatter([],[],c=PREY,s=6,edgecolors="none",zorder=3)
sp4r=ax4a.scatter([],[],c=PRED,s=90,zorder=5,edgecolors="#880000",lw=0.5,marker="o")
rint4=ax4a.text(0.03,0.97,"",transform=ax4a.transAxes,color="#333333",fontsize=8,va="top")
ax4b.set_xlim(0,2.5); ax4b.set_ylim(0,125)
ax4b.set_xlabel("Interaction Radius R_int"); ax4b.set_ylabel("N survived")
ax4b.set_title("Survival vs R_int  (N=120)",fontsize=9)
ax4b.axhline(120,color="#aaaaaa",lw=0.8,ls="--"); ax4b.grid(True,alpha=0.3)
ax4b.axvspan(0.35,1.3,alpha=0.07,color="green")
lines4={}
for idx,delta in enumerate(delta_list4):
    ln,=ax4b.plot([],[],"-o",color=clrs4[idx],label=f"delta={delta}",lw=2,ms=4)
    lines4[delta]=ln
ax4b.legend(fontsize=9,frameon=True,edgecolor="#cccccc")

def upd4(frame):
    fi=min(frame,len(live4)-1); x,y,xp,yp,alive=live4[fi]
    if alive.any():
        pts_x=np.append(x[alive],xp); pts_y=np.append(y[alive],yp)
        xl,xr,yl,yr=dyn_lims(pts_x,pts_y)
        ax4a.set_xlim(xl,xr); ax4a.set_ylim(yl,yr)
        sp4p.set_offsets(np.column_stack([x[alive],y[alive]]))
    sp4r.set_offsets([[xp,yp]]); rint4.set_text(f"N={alive.sum()}")
    ri=min(frame//2,len(R_scan4)-1)
    for delta in delta_list4: lines4[delta].set_data(R_scan4[:ri+1],Nsur4[delta][:ri+1])
    return [sp4p,sp4r,rint4]+list(lines4.values())

nf4=max(200,len(R_scan4)*2)
ani4=FuncAnimation(fig4,upd4,frames=nf4,interval=70,blit=False)
ani4.save("animations/gif4_survival_vs_rint.gif",writer=WRITER)
plt.close(fig4)
print("  saved gif4_survival_vs_rint.gif")


# ── GIF 5: Weak predator ring vs chasing ─────────────────────────────────────
print("[GIF 5] Weak predator ring vs chasing...")
cases5=[dict(M_pd=2.0,label="M_pd=2.0  Stable Ring"),
        dict(M_pd=3.5,label="M_pd=3.5  Chasing")]
NF5=250; SPF5=6
all_g5=[(c,)+sim_inertial(80,1.,1.,0.2,0.4,1.,c["M_pd"],NF5,SPF5,dt=0.02,seed=42)
        for c in cases5]
for c in cases5: print(f"  M_pd={c['M_pd']} done")

fig5,(ax5a,ax5b)=plt.subplots(1,2,figsize=(10,4.8))
fig5.suptitle("Weak Predator (delta'=0.4, M_pr=1): Ring Formation vs Chasing\n"
              "Chakraborty, Laha and De (2022)",fontsize=10,fontweight="bold")
fig5.subplots_adjust(wspace=0.3,top=0.82,bottom=0.22)
axes5g=[ax5a,ax5b]; s5p=[]; s5r=[]; ph5ln=[]; ph5h=[[],[]]; quiv5=[None,None]

for k,(cfg,frms,phis) in enumerate(all_g5):
    ax=axes5g[k]; ax.set_aspect("equal"); ax.grid(True,alpha=0.2)
    ax.set_title(cfg["label"],fontsize=8.5); ax.set_xlabel("X"); ax.set_ylabel("Y")
    x0,y0,vx0,vy0,xp0,yp0,al0=frms[0]
    s1=ax.scatter(x0[al0],y0[al0],c=PREY,s=6,edgecolors="none",zorder=3)
    s2=ax.scatter([xp0],[yp0],c=PRED,s=90,zorder=5,edgecolors="#880000",lw=0.8,marker="o")
    s5p.append(s1); s5r.append(s2)
    phi_ax=ax.inset_axes([0.0,-0.32,1.0,0.24])
    phi_ax.set_facecolor("white"); phi_ax.set_xlim(0,NF5); phi_ax.set_ylim(-1.1,1.1)
    phi_ax.axhline(0,color="#aaaaaa",lw=0.6,ls="--"); phi_ax.grid(True,alpha=0.2)
    phi_ax.set_xlabel("Frame",fontsize=6.5); phi_ax.set_ylabel("phi",fontsize=6.5)
    phi_ax.tick_params(labelsize=5.5,direction="in")
    ln,=phi_ax.plot([],[],color="#226622",lw=1.3); ph5ln.append(ln)
    for sp in ax.spines.values(): sp.set_linewidth(0.7)

def upd5(frame):
    arts=[]
    for k,(cfg,frms,phis) in enumerate(all_g5):
        fi=min(frame,len(frms)-1); x,y,vx,vy,xp,yp,alive=frms[fi]
        ax=axes5g[k]
        if alive.any():
            pts_x=np.append(x[alive],xp); pts_y=np.append(y[alive],yp)
            xl,xr,yl,yr=dyn_lims(pts_x,pts_y)
            ax.set_xlim(xl,xr); ax.set_ylim(yl,yr)
            s5p[k].set_offsets(np.column_stack([x[alive],y[alive]]))
            if quiv5[k] is not None: quiv5[k].remove()
            norm=np.maximum(np.sqrt(vx[alive]**2+vy[alive]**2),1e-12)
            sc=2*max(xr-xl,yr-yl)*0.035
            quiv5[k]=ax.quiver(x[alive],y[alive],vx[alive]/norm*sc,vy[alive]/norm*sc,
                               color=VEL,alpha=0.45,scale_units="xy",scale=1,
                               width=0.0025,headwidth=3.5,headlength=3.5)
        s5r[k].set_offsets([[xp,yp]])
        if fi>0 and fi-1<len(phis): ph5h[k].append(phis[fi-1])
        ph5ln[k].set_data(range(len(ph5h[k])),ph5h[k])
        arts+=[s5p[k],s5r[k],ph5ln[k]]
        if quiv5[k] is not None: arts.append(quiv5[k])
    return arts

ani5=FuncAnimation(fig5,upd5,frames=NF5+1,interval=60,blit=False)
ani5.save("animations/gif5_weak_predator_ring.gif",writer=WRITER)
plt.close(fig5)
print("  saved gif5_weak_predator_ring.gif")


# ── GIF 6: Strong predator 3 masses ──────────────────────────────────────────
print("[GIF 6] Strong predator 3 masses...")
cases6=[dict(M_pd=0.1, label="M_pd=0.1  Light: captures all"),
        dict(M_pd=1.0, label="M_pd=1.0  Medium: splitting"),
        dict(M_pd=100.,label="M_pd=100  Heavy: F-maneuver")]
NF6=250; SPF6=5
all_g6=[(c,)+sim_inertial(80,1.,1.,0.2,2.5,1.,c["M_pd"],NF6,SPF6,dt=0.02,seed=99)
        for c in cases6]
for c in cases6: print(f"  M_pd={c['M_pd']} done")

fig6,axes6=plt.subplots(1,3,figsize=(14,4.8))
fig6.suptitle("Strong Predator (delta'=2.5, M_pr=1): Escape Trajectories vs M_pd\n"
              "Chakraborty, Laha and De (2022)",fontsize=10,fontweight="bold")
fig6.subplots_adjust(wspace=0.32,top=0.82,bottom=0.12,left=0.06,right=0.97)
s6p=[]; s6d=[]; s6r=[]; trl6l=[]; sur6t=[]
trail6h=[[] for _ in range(3)]; quiv6=[None,None,None]; TRAIL=35

for k,(cfg,frms,phis) in enumerate(all_g6):
    ax=axes6[k]; ax.set_aspect("equal"); ax.grid(True,alpha=0.2)
    ax.set_title(cfg["label"],fontsize=8.5); ax.set_xlabel("X"); ax.set_ylabel("Y")
    x0,y0,vx0,vy0,xp0,yp0,al0=frms[0]
    s1=ax.scatter(x0[al0],y0[al0],c=PREY,s=6,edgecolors="none",zorder=3)
    s2=ax.scatter([],[],c=DEAD,s=5,edgecolors="none",alpha=0.4,zorder=2)
    s3=ax.scatter([xp0],[yp0],c=PRED,s=90,zorder=5,edgecolors="#880000",lw=0.8,marker="o")
    tl,=ax.plot([],[],color=TCLR,lw=1.0,alpha=0.5,zorder=4)
    st=ax.text(0.03,0.97,"N=80",transform=ax.transAxes,color="#333333",fontsize=7.5,va="top")
    for sp in ax.spines.values(): sp.set_linewidth(0.7)
    s6p.append(s1); s6d.append(s2); s6r.append(s3); trl6l.append(tl); sur6t.append(st)

def upd6(frame):
    arts=[]
    for k,(cfg,frms,phis) in enumerate(all_g6):
        fi=min(frame,len(frms)-1); x,y,vx,vy,xp,yp,alive=frms[fi]; ax=axes6[k]
        if alive.any():
            pts_x=np.append(x[alive],xp); pts_y=np.append(y[alive],yp)
            xl,xr,yl,yr=dyn_lims(pts_x,pts_y,pad=0.18)
            ax.set_xlim(xl,xr); ax.set_ylim(yl,yr)
            s6p[k].set_offsets(np.column_stack([x[alive],y[alive]]))
            if quiv6[k] is not None: quiv6[k].remove()
            norm=np.maximum(np.sqrt(vx[alive]**2+vy[alive]**2),1e-12)
            sc=2*max(xr-xl,yr-yl)*0.035
            quiv6[k]=ax.quiver(x[alive],y[alive],vx[alive]/norm*sc,vy[alive]/norm*sc,
                               color=VEL,alpha=0.4,scale_units="xy",scale=1,
                               width=0.0025,headwidth=3.5,headlength=3.5)
        else: s6p[k].set_offsets(np.empty((0,2)))
        if (~alive).any():
            s6d[k].set_offsets(np.column_stack([x[~alive],y[~alive]]))
        s6r[k].set_offsets([[xp,yp]])
        trail6h[k].append((xp,yp)); trail6h[k]=trail6h[k][-TRAIL:]
        tr=np.array(trail6h[k]); trl6l[k].set_data(tr[:,0],tr[:,1])
        sur6t[k].set_text(f"N={alive.sum()}")
        arts+=[s6p[k],s6d[k],s6r[k],trl6l[k],sur6t[k]]
        if quiv6[k] is not None: arts.append(quiv6[k])
    return arts

ani6=FuncAnimation(fig6,upd6,frames=NF6+1,interval=55,blit=False)
ani6.save("animations/gif6_strong_predator_escape.gif",writer=WRITER)
plt.close(fig6)
print("  saved gif6_strong_predator_escape.gif")


# ── GIF 7: phi(T) 4 panels ───────────────────────────────────────────────────
print("[GIF 7] phi(T) 4-panel live build...")
Mpd7=[0.1,1.0,3.0,100.0]
labels7=["(a) M_pd=0.1  light","(b) M_pd=1.0  medium",
         "(c) M_pd=3.0  heavier","(d) M_pd=100  heavy"]
clrs7=["#cc2222","#cc6600","#1a5ea8","#228833"]
NF7=250; SPF7=8
all_g7=[(Mpd,)+sim_inertial(60,1.,1.,0.2,2.5,1.,Mpd,NF7,SPF7,dt=0.02,seed=42)
        for Mpd in Mpd7]
for Mpd in Mpd7: print(f"  M_pd={Mpd} done")

fig7,axes7=plt.subplots(2,2,figsize=(10,7)); axes7=axes7.ravel()
fig7.suptitle("Order Parameter phi(T): Prey-Predator Direction Alignment\n"
              "Chakraborty, Laha and De (2022) Fig 5",fontsize=11,fontweight="bold")
fig7.subplots_adjust(hspace=0.45,wspace=0.3,top=0.88,bottom=0.08,left=0.1,right=0.97)
ph7l=[]; ph7pt=[]; ph7h=[[] for _ in range(4)]; T_end=NF7*SPF7*0.02

for k,(Mpd,frms,phis) in enumerate(all_g7):
    ax=axes7[k]; ax.set_xlim(0,T_end); ax.set_ylim(-1.25,1.25)
    ax.axhline(0,color="#888888",lw=0.8,ls="--")
    ax.fill_between([0,T_end],[0],[1.1],alpha=0.05,color="green")
    ax.fill_between([0,T_end],[-1.1],[0],alpha=0.05,color="red")
    ax.tick_params(direction="in"); ax.grid(True,alpha=0.25)
    ax.set_title(labels7[k],fontsize=9); ax.set_xlabel("Time T"); ax.set_ylabel("phi")
    ax.text(0.02,0.93,"together",transform=ax.transAxes,
            fontsize=7.5,color="#226622",va="top")
    ax.text(0.02,0.07,"apart",transform=ax.transAxes,
            fontsize=7.5,color="#882222",va="bottom")
    for sp in ax.spines.values(): sp.set_linewidth(0.7)
    ln,=ax.plot([],[],color=clrs7[k],lw=1.8); ph7l.append(ln)
    pt,=ax.plot([],[],"o",color="#333333",ms=5,zorder=5); ph7pt.append(pt)

def upd7(frame):
    arts=[]
    for k,(Mpd,frms,phis) in enumerate(all_g7):
        fi=min(frame,len(phis)-1); ph7h[k].append(np.clip(phis[fi],-1,1))
        T_vals=[i*SPF7*0.02 for i in range(len(ph7h[k]))]
        ph7l[k].set_data(T_vals,ph7h[k]); ph7pt[k].set_data([T_vals[-1]],[ph7h[k][-1]])
        arts+=[ph7l[k],ph7pt[k]]
    return arts

ani7=FuncAnimation(fig7,upd7,frames=NF7,interval=55,blit=True)
ani7.save("animations/gif7_phi_timeseries.gif",writer=WRITER)
plt.close(fig7)
print("  saved gif7_phi_timeseries.gif")


# ── GIF 8: Phase diagram M_pd sweep ──────────────────────────────────────────
print("[GIF 8] Phase diagram sweep...")
Mpd_sweep=np.logspace(-1,2,28)

def sim_final(Mpd,N=60,seed=3):
    rng=np.random.RandomState(seed)
    x=rng.random(N); y=rng.random(N); vx=np.zeros(N); vy=np.zeros(N)
    xp=1.5; yp=0.5; vxp=0.; vyp=0.; alive=np.ones(N,dtype=bool)
    for _ in range(400):
        x,y,vx,vy,xp,yp,vxp,vyp,alive=inertial_step(
            x,y,vx,vy,xp,yp,vxp,vyp,alive,1.,1.,0.2,2.5,1.,Mpd,0.02)
    return alive.sum()

Nsur8=np.array([np.mean([sim_final(M,seed=s*5+3) for s in range(3)])
                for M in Mpd_sweep])
print("  survival grid done")

NF8=200; SPF8=4
live_light,_=sim_inertial(60,1.,1.,0.2,2.5,1.,0.1, NF8,SPF8,seed=7)
live_heavy,_=sim_inertial(60,1.,1.,0.2,2.5,1.,100.,NF8,SPF8,seed=7)

fig8,(ax8a,ax8b)=plt.subplots(1,2,figsize=(12,4.8))
fig8.suptitle("Inertial Model: Survival Phase Diagram\n"
              "Three Regimes: Killed / Competitive / Survival",
              fontsize=10,fontweight="bold")
fig8.subplots_adjust(wspace=0.35,top=0.83,bottom=0.14,left=0.07,right=0.97)
ax8a.set_aspect("equal"); ax8a.grid(True,alpha=0.2)
ax8a.set_xlabel("X"); ax8a.set_ylabel("Y"); ax8a.set_title("Live Simulation",fontsize=9)
sp8p=ax8a.scatter([],[],c=PREY,s=6,edgecolors="none",zorder=3)
sp8d=ax8a.scatter([],[],c=DEAD,s=5,edgecolors="none",alpha=0.4)
sp8r=ax8a.scatter([],[],c=PRED,s=90,zorder=5,edgecolors="#880000",lw=0.7,marker="o")
trail8=[]; trail8l,=ax8a.plot([],[],color=TCLR,lw=1.,alpha=0.5)
mode8=ax8a.text(0.03,0.97,"",transform=ax8a.transAxes,color="#333333",fontsize=8,va="top")
sur8 =ax8a.text(0.03,0.87,"",transform=ax8a.transAxes,color="#333333",fontsize=8,va="top")

ax8b.set_xscale("log"); ax8b.set_xlim(0.08,130); ax8b.set_ylim(-2,65)
ax8b.axvspan(0.08,0.5,alpha=0.10,color="#cc3333")
ax8b.axvspan(0.5, 8., alpha=0.10,color="#cc7700")
ax8b.axvspan(8., 130, alpha=0.10,color="#226622")
ax8b.text(0.10,59,"Killed",color="#aa1111",fontsize=9,fontweight="bold")
ax8b.text(0.65,59,"Competitive",color="#996600",fontsize=9,fontweight="bold")
ax8b.text(12,  59,"Survival",color="#116611",fontsize=9,fontweight="bold")
ax8b.set_xlabel("M_pd  (predator mass)"); ax8b.set_ylabel("N survived  (out of 60)")
ax8b.set_title("Survival vs Predator Mass  (delta'=2.5)",fontsize=9)
ax8b.grid(True,alpha=0.25,which="both")
phase_ln,=ax8b.plot([],[],"-o",color="#1a5ea8",lw=2,ms=6)
cur_pt,  =ax8b.plot([],[],"o",color="#cc2222",ms=11,zorder=6)
vcursor  =ax8b.axvline(Mpd_sweep[0],color="#cc2222",lw=1.1,ls="--",alpha=0.6)
half_pt=NF8//2

def upd8(frame):
    idx=min(frame,len(Mpd_sweep)-1); fi=min(frame*2,NF8)
    if frame<half_pt:
        x,y,vx,vy,xp,yp,alive=live_light[min(fi,len(live_light)-1)]
        mode8.set_text("M_pd=0.1  (light)")
    else:
        fi2=min((frame-half_pt)*2,NF8)
        x,y,vx,vy,xp,yp,alive=live_heavy[min(fi2,len(live_heavy)-1)]
        mode8.set_text("M_pd=100  (heavy)")
    if alive.any():
        pts_x=np.append(x[alive],xp); pts_y=np.append(y[alive],yp)
        xl,xr,yl,yr=dyn_lims(pts_x,pts_y)
        ax8a.set_xlim(xl,xr); ax8a.set_ylim(yl,yr)
        sp8p.set_offsets(np.column_stack([x[alive],y[alive]]))
    else: sp8p.set_offsets(np.empty((0,2)))
    if (~alive).any():
        sp8d.set_offsets(np.column_stack([x[~alive],y[~alive]]))
    sp8r.set_offsets([[xp,yp]])
    trail8.append((xp,yp))
    if len(trail8)>30: trail8.pop(0)
    tr=np.array(trail8); trail8l.set_data(tr[:,0],tr[:,1])
    sur8.set_text(f"N={alive.sum()}")
    phase_ln.set_data(Mpd_sweep[:idx+1],Nsur8[:idx+1])
    cur_pt.set_data([Mpd_sweep[idx]],[Nsur8[idx]])
    vcursor.set_xdata([Mpd_sweep[idx],Mpd_sweep[idx]])
    return sp8p,sp8d,sp8r,trail8l,mode8,sur8,phase_ln,cur_pt

ani8=FuncAnimation(fig8,upd8,frames=len(Mpd_sweep),interval=130,blit=False)
ani8.save("animations/gif8_phase_diagram.gif",writer=WRITER)
plt.close(fig8)
print("  saved gif8_phase_diagram.gif")

print("\n" + "="*60)
print("All 8 GIFs saved to ./animations/")
print("="*60)
