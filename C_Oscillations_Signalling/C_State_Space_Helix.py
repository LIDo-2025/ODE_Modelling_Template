"""
C · Enzyme Oscillations — Phase Portrait & Spatio-temporal Helix
=================================================================
Two-panel animated visualisation of the enzyme oscillator.

Left — Phase portrait  (P horizontal, S vertical)
    S upward matches the time-series convention: the orbit is literally the
    "breathing" S(t) curve wrapped into a closed loop.  The open white circle
    marks the unstable fixed point — the organising centre the limit cycle
    encircles.  Nullclines and vector field show the forces driving the orbit.

Right — Spatio-temporal helix  (Time, P, S)
    Opens as a standard time series: time flows left→right, S oscillates up/down.
    Grab and rotate to reveal the helical 3-D structure.  Projections:
      · back wall  (y = P_hi)  →  S(t) time series        [sky blue]
      · floor      (z = S_lo)  →  P(t) time series        [rose]
      · left face  (x = 0)     →  S vs P  =  limit cycle  [turquoise]

Model: feedback-inhibited enzyme
    dS/dt = a1 − b1·S − v(S,P)     v = k_max·S / (K_m + S³) / (1 + k_i·P^2.8)
    dP/dt = a2 − b2·P + v(S,P)
    Oscillatory window: a1 ∈ [0.63, 0.72]  (period ≈ 68 time units at a1=0.67)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ── Colour theme ──────────────────────────────────────────────────────────────
BG_FIG   = '#0a0a0a'   # figure background (near-black)
BG_AXES  = '#080814'   # left panel axes background (dark)
BG_3D    = '#0d0d1a'   # 3D pane faces (near-black — matches figure background)
CLR_TXT  = '#d8d8f0'
CLR_S_NC  = '#38bdf8'   # S-nullcline / S(t) projection  (sky blue)
CLR_P_NC  = '#f87171'   # P-nullcline / P(t) projection  (rose)
CLR_ORBIT = '#00e5cf'   # limit cycle + helix  (turquoise — contrasts both backgrounds)
CMAP      = plt.cm.plasma

# ── Model ─────────────────────────────────────────────────────────────────────
PAR = dict(a1=0.67, b1=0.18, a2=0.02, b2=0.05,
           k_max=25.0, K_m=0.34, k_i=0.06, n=1, m=3, q=2.8)


def rhs(t, y):
    S  = max(y[0], 1e-9)
    Pv = max(y[1], 1e-9)
    rate = (PAR['k_max'] * S**PAR['n']) \
         / (PAR['K_m']   + S**PAR['m']) \
         / (1.0 + PAR['k_i'] * Pv**PAR['q'])
    return [PAR['a1'] - PAR['b1'] * S  - rate,
            PAR['a2'] - PAR['b2'] * Pv + rate]


# ── Simulate ──────────────────────────────────────────────────────────────────
print("Simulating enzyme oscillator …")
T_TRANS = 400
T_SHOW  = 210      # ≈ 3 complete cycles  (period ≈ 68 time units)
N_PTS   = 700

sol = solve_ivp(rhs, [0, T_TRANS + T_SHOW], [2.5, 0.6],
                method='LSODA', dense_output=True, rtol=1e-9, atol=1e-11)

t_eval = np.linspace(T_TRANS, T_TRANS + T_SHOW, N_PTS)
traj   = sol.sol(t_eval)
S_tr   = traj[0]
P_tr   = traj[1]
t_rel  = t_eval - t_eval[0]

S_lo, S_hi = 0.0,              S_tr.max() * 1.22
P_lo, P_hi = P_tr.min() * 0.80, P_tr.max() * 1.10


# ── Unstable fixed point ──────────────────────────────────────────────────────
fp_sol, _, fp_ier, _ = fsolve(lambda y: rhs(0, y), [1.4, 9.0], full_output=True)
fp = fp_sol if fp_ier == 1 else None


# ── Vector field ──────────────────────────────────────────────────────────────
# Grid is computed with SG (S along cols) and PG (P along rows).
# For the swapped left panel (x=P, y=S) we need .T on all 2-D arrays.
print("Computing vector field …")
NG     = 36
sg     = np.linspace(S_lo, S_hi, NG)   # S values
pg     = np.linspace(P_lo, P_hi, NG)   # P values
SG, PG = np.meshgrid(sg, pg)           # SG[i,j]=sg[j], PG[i,j]=pg[i]

DS = np.zeros_like(SG)
DP = np.zeros_like(PG)
for i in range(NG):
    for j in range(NG):
        ds, dp = rhs(0, [SG[i, j], PG[i, j]])
        DS[i, j] = ds
        DP[i, j] = dp

SPEED = np.sqrt(DS**2 + DP**2)
DS_n  = DS / (SPEED + 1e-9)
DP_n  = DP / (SPEED + 1e-9)

# Transposed versions for the swapped axes (x=P, y=S):
# DS_n.T[row_S, col_P] = dS/dt at S=sg[row_S], P=pg[col_P]  →  y-component
# DP_n.T[row_S, col_P] = dP/dt at S=sg[row_S], P=pg[col_P]  →  x-component
DS_nT  = DS_n.T
DP_nT  = DP_n.T
SPEEDT = SPEED.T


# ── Figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({'text.color':      CLR_TXT,
                     'axes.labelcolor': CLR_TXT,
                     'xtick.color':     CLR_TXT,
                     'ytick.color':     CLR_TXT})

fig = plt.figure(figsize=(14, 6.5))
fig.patch.set_facecolor(BG_FIG)
plt.subplots_adjust(left=0.07, right=0.97,
                    bottom=0.10, top=0.92, wspace=0.14)


# ── Left panel: phase portrait  (x = P,  y = S  ↑) ───────────────────────────
ax2 = fig.add_subplot(1, 2, 1)
ax2.set_facecolor(BG_AXES)
ax2.set_xlabel('Product  P',   fontsize=12)
ax2.set_ylabel('Substrate  S', fontsize=12)    # S is now the vertical axis
ax2.set_title('Phase Portrait  (P, S)', fontsize=13, pad=10)
ax2.set_xlim(P_lo, P_hi)
ax2.set_ylim(S_lo, S_hi)
ax2.tick_params(colors=CLR_TXT)
for sp in ax2.spines.values():
    sp.set_edgecolor('#333355')
ax2.grid(True, alpha=0.07, color='white', linestyle='--')

# Streamlines — horizontal component = dP/dt, vertical component = dS/dt
ax2.streamplot(pg, sg, DP_nT, DS_nT,
               color=np.log1p(SPEEDT),
               cmap='plasma', density=1.4,
               linewidth=0.9, arrowsize=0.8, zorder=2)

# Nullclines: contour on the swapped (P, S) grid
#   PG.T[row_S, col_P] = pg[col_P]  →  x-coordinate ✓
#   SG.T[row_S, col_P] = sg[row_S]  →  y-coordinate ✓
ax2.contour(PG.T, SG.T, DS.T, levels=[0],
            colors=[CLR_S_NC], linewidths=2.0, zorder=3)
ax2.contour(PG.T, SG.T, DP.T, levels=[0],
            colors=[CLR_P_NC], linewidths=2.0, zorder=3)

# Ghost limit-cycle outline
ax2.plot(P_tr, S_tr, color=CLR_ORBIT, lw=1.8, alpha=0.85, zorder=4)

# Unstable fixed point — open circle (hollow = unstable)
legend_handles = [
    Line2D([0], [0], color=CLR_S_NC, lw=2, label='S-nullcline  (dS/dt = 0)'),
    Line2D([0], [0], color=CLR_P_NC, lw=2, label='P-nullcline  (dP/dt = 0)'),
]
if fp is not None and P_lo <= fp[1] <= P_hi and S_lo <= fp[0] <= S_hi:
    ax2.plot(fp[1], fp[0], 'o',            # x=P_fp, y=S_fp
             color='white', ms=10,
             markerfacecolor='none',
             markeredgewidth=2.0,
             markeredgecolor='white',
             zorder=7)
    legend_handles.append(
        Line2D([0], [0], marker='o', color='w', ms=8,
               markerfacecolor='none', markeredgewidth=2,
               markeredgecolor='white', linestyle='none',
               label='Unstable fixed point'))

ax2.legend(handles=legend_handles, fontsize=9, loc='upper right',
           framealpha=0.45, facecolor='#1a1a2e', edgecolor='#444466')

# Animated comet (2D LineCollection) + double glow + head
comet2 = LineCollection([], linewidth=2.2, zorder=5)
ax2.add_collection(comet2)
glow_o, = ax2.plot([], [], 'o', color=CLR_ORBIT, ms=22, alpha=0.05, zorder=6)
glow_i, = ax2.plot([], [], 'o', color=CLR_ORBIT, ms=12, alpha=0.15, zorder=7)
head2,  = ax2.plot([], [], 'o', color=CLR_ORBIT, ms=5,  alpha=1.00, zorder=8)


# ── Right panel: 3D spatio-temporal helix ─────────────────────────────────────
# Axes: x = Time,  y = P,  z = S
# azim = -90  →  camera on the -y side, looking toward +y
#              x (Time) increases to the RIGHT  ✓
#              z (S)    increases UPWARD         ✓
ax3 = fig.add_subplot(1, 2, 2, projection='3d')

# Kill the white outer region of the 3D axes widget
ax3.patch.set_facecolor(BG_FIG)
ax3.patch.set_alpha(1.0)

# Dark pane faces
for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
    pane.fill = True
    pane.set_facecolor(BG_3D)
    pane.set_edgecolor('#2a2a3a')
try:
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis._axinfo['grid']['color'] = '#3a3a50'
except Exception:
    pass

ax3.set_xlabel('Time',         fontsize=10, labelpad=6)
ax3.set_ylabel('Product  P',   fontsize=10, labelpad=6)
ax3.set_zlabel('Substrate  S', fontsize=10, labelpad=6)
ax3.set_title('Spatio-temporal Helix', fontsize=13, pad=12)
ax3.set_xlim(0,    t_rel.max())
ax3.set_ylim(P_lo, P_hi)
ax3.set_zlim(S_lo, S_hi)

# Initial view: time series  (time →,  S ↑,  P into screen)
ax3.view_init(elev=5, azim=-90)

# 3D axis label and tick colours — tick_params is reliable; label loop handles axis titles
ax3.tick_params(axis='x', colors='white')
ax3.tick_params(axis='y', colors='white')
ax3.tick_params(axis='z', colors='white')
for label in (ax3.xaxis.label, ax3.yaxis.label, ax3.zaxis.label):
    label.set_color('white')

# Static helix — uniform turquoise so it reads as the same object as the left-panel orbit
pts3d  = np.array([t_rel, P_tr, S_tr]).T.reshape(-1, 1, 3)
segs3d = np.concatenate([pts3d[:-1], pts3d[1:]], axis=1)
helix_lc = Line3DCollection(segs3d, color=CLR_ORBIT, linewidth=1.8, alpha=0.75)
ax3.add_collection3d(helix_lc)

# Left face (x=0): limit-cycle shadow  (S vs P) — dim turquoise echo
ax3.plot(np.zeros_like(P_tr), P_tr, S_tr,
         color=CLR_ORBIT, lw=0.8, alpha=0.30, zorder=1)

# Back wall (y=P_hi): S(t) time series shadow — bright enough to read on black
ax3.plot(t_rel, np.full_like(P_tr, P_hi), S_tr,
         color='#c8c8e0', lw=1.2, alpha=0.80, zorder=1)

# Floor (z=S_lo): P(t) time series shadow — bright enough to read on black
ax3.plot(t_rel, P_tr, np.full_like(S_tr, S_lo),
         color='#c8c8e0', lw=1.2, alpha=0.80, zorder=1)

# Animated tail + head
tail3, = ax3.plot([], [], [], '-', color=CLR_ORBIT, lw=2.5, alpha=0.85, zorder=5)
head3, = ax3.plot([], [], [], 'o', color=CLR_ORBIT, ms=7,  alpha=1.00, zorder=6)


# ── Animation ─────────────────────────────────────────────────────────────────
TAIL  = 55
frame = [0]


def comet_segments(xs, ys):
    """2-D RGBA LineCollection segments: transparent tail → bright head."""
    if len(xs) < 2:
        return [], []
    segs   = [np.array([[xs[k], ys[k]], [xs[k+1], ys[k+1]]])
              for k in range(len(xs) - 1)]
    alphas = np.linspace(0.03, 0.92, len(segs))
    return segs, [(0.0, 0.898, 0.812, a) for a in alphas]   # CLR_ORBIT = #00e5cf


def update(_=None):
    i  = frame[0]
    lo = max(0, i - TAIL)

    # Left panel — x=P, y=S
    segs, cols = comet_segments(P_tr[lo: i+1], S_tr[lo: i+1])
    comet2.set_segments(segs)
    comet2.set_color(cols)
    glow_o.set_data([P_tr[i]], [S_tr[i]])
    glow_i.set_data([P_tr[i]], [S_tr[i]])
    head2.set_data( [P_tr[i]], [S_tr[i]])

    # Right panel — x=time, y=P, z=S
    tail3.set_data(t_rel[lo: i+1], P_tr[lo: i+1])
    tail3.set_3d_properties(S_tr[lo: i+1])
    head3.set_data([t_rel[i]], [P_tr[i]])
    head3.set_3d_properties([S_tr[i]])

    frame[0] = (i + 1) % len(S_tr)
    fig.canvas.draw_idle()


update()
timer = fig.canvas.new_timer(interval=30)
timer.add_callback(update)
timer.start()

print("Period ≈ 68 time units | 2 cycles shown | rotate the right panel to explore.")
plt.show()
